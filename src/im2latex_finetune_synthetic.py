import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoFeatureExtractor,
    get_linear_schedule_with_warmup
)

from peft import LoraConfig, get_peft_model

from PIL import Image
import evaluate
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm
import os
import json
import time

from train_config import Config

import warnings
warnings.filterwarnings("ignore")

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

# distributed process group initialization
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = ddp_rank == 0

torch.set_float32_matmul_precision(Config.float32_matmul_precision)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# setting a seed for reproducibility
set_seed(Config.seed)

# getting pre-trained tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
feature_extractor = AutoFeatureExtractor.from_pretrained(Config.feature_extractor)

# loading new dataset
new_dataset = load_dataset("linxy/LaTeX_OCR", "synthetic_handwrite")

# combining train, val, and test splits and then splitting into train and val
combined_ds = new_dataset['train'].train_test_split(test_size=0.1, seed=Config.seed)
train_ds = combined_ds['train']
val_ds = combined_ds['test']

def filter_dataset(dataset):
    def is_valid_sample(sample):
        try:
            image = sample['image']
            latex = sample['text']
            return image is not None and latex is not None and len(latex) > 0
        except:
            return False

    return dataset.filter(is_valid_sample)

train_ds = filter_dataset(train_ds)
val_ds = filter_dataset(val_ds)

if master_process:
    print("Length of train set after splitting:", len(train_ds))
    print("Length of val set after splitting:", len(val_ds))

# setting up the model
model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex").to(device)

# setting up the adapter
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        # Encoder (Swin Transformer) modules
        "attn.qkv",
        "attn.proj",
        "mlp.fc1",
        "mlp.fc2",
        # Decoder (GPT-2) modules
        "c_attn",
        "c_proj",
        "c_fc",
        "attn.c_proj",
    ],
    lora_dropout=0.1,
    bias="none"
    # task_type="VL"  # Vision-Language task
)

# applying lora
model = get_peft_model(model, lora_config)
if master_process:  
    model.print_trainable_parameters() 

torch.compile(model)
model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank, find_unused_parameters=False)

# dataset loading class
class LatexDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        feature_extractor,
        phase,
        image_size=Config.image_size,
        max_length=512
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.phase = phase
        self.image_size = image_size
        self.max_length = max_length
        self.train_transform = self.get_train_transform()

    def __len__(self):
        return len(self.dataset)

    def get_train_transform(self):
        def train_transform(image):
            image = image.resize(self.image_size)
            image = np.array(image)
            image = image.astype(np.float32) / 255.0
            return image
        return train_transform

    def __getitem__(self, idx):
        item = self.dataset[idx]
        latex_sequence = item['text']
        image = item['image']

        # converting RGBA to RGB for the test set --> some images have alphas
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # image processing
        try:
            pixel_values = self.feature_extractor(
                images=image.resize(self.image_size),
                return_tensors="pt",
            ).pixel_values.squeeze()
            if pixel_values.ndim == 0:
                raise ValueError("Processed image has no dimensions")
        except Exception as e:
            print(f"Error processing image at index {idx}: {str(e)}")
            # provide a default tensor in case of error
            pixel_values = torch.zeros((3, self.image_size[0], self.image_size[1]))

        # tokenization
        try:
            latex_tokens = self.tokenizer(
                latex_sequence,
                padding=False,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            ).input_ids.squeeze()
            if latex_tokens.ndim == 0:
                raise ValueError("Tokenized latex has no dimensions")
        except Exception as e:
            print(f"Error tokenizing latex at index {idx}: {str(e)}")
            # provide a default tensor in case of error
            latex_tokens = torch.zeros(1, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": latex_tokens
        }

# custom data collator
def data_collator(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Handle labels, ensuring it's always a list of tensors
    labels = [item['labels'] for item in batch if item['labels'].numel() > 0]
    
    if len(labels) == 0:
        # if all labels are empty, return a dummy tensor
        labels = torch.zeros((len(batch), 1), dtype=torch.long)
    elif len(labels) == 1:
        # if there's only one sample, add a dimension to make it a batch
        labels = labels[0].unsqueeze(0)
    else:
        # for multiple samples, use pad_sequence as before
        labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }

# creating datasets and dataloader
train_dataset = LatexDataset(train_ds, tokenizer, feature_extractor, phase='train')
val_dataset = LatexDataset(val_ds, tokenizer, feature_extractor, phase='val')

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset, shuffle=False)

train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size_train, sampler=train_sampler, collate_fn=data_collator, drop_last=True )
val_dataloader = DataLoader(val_dataset, batch_size=Config.batch_size_val, sampler=val_sampler, collate_fn=data_collator, drop_last=True )

# training parameters
learning_rate = Config.learning_rate
num_epochs = Config.num_epochs  # using epochs for printing purposes actually, but control by max_steps
warmup_steps = Config.warmup_steps
eval_steps = 100

# effective batch size per GPU (or per process)
effective_batch_size = Config.batch_size_train * ddp_world_size
if master_process:
    print("Effective batch size:", effective_batch_size)

# calculate max_steps
max_steps = (len(train_dataset) // effective_batch_size) * num_epochs
if master_process:
    print("Max steps:", max_steps)

# initializing optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=Config.betas, eps=Config.eps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=max_steps
)

# metric for val purposes
bleu_metric = evaluate.load(Config.bleu)

best_checkpoint_step = None

# training loop for LoRA fine-tuning
def train_lora(model, train_dataloader, optimizer, scheduler, device, num_epochs, eval_steps, val_dataloader, tokenizer, bleu_metric, local_rank=0):
    model.train()
    train_losses = [] # list to store losses for whole epoch averaging
    interval_losses = [] # list to store interval losses (updates every eval_steps)
    best_val_loss = float('inf')
    all_metrics = [] # list to store all metrics

    checkpoint_dir = Config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # calculating total steps per epoch
    total_steps_per_epoch = len(train_dataloader)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=local_rank != 0)):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # averaging losses over gpus
            interval_loss_tensor = loss.clone().detach().to(device)
            torch.distributed.all_reduce(interval_loss_tensor, op=torch.distributed.ReduceOp.AVG)
            interval_losses.append(interval_loss_tensor.item())

            # Increment global_step
            global_step = epoch * total_steps_per_epoch + step + 1

            # logging and averaging loss every eval_steps
            if global_step % eval_steps == 0 or (epoch == num_epochs - 1 and step == total_steps_per_epoch - 1):
                # computing the average loss for the last eval_steps
                average_loss = np.mean(interval_losses)
                train_losses.append(average_loss)
                if master_process:
                    print(" ")
                    print("-----------------------------------------------------------")
                    print(f"Step {global_step} - Average Training Loss: {average_loss}")
                    print("-----------------------------------------------------------")

                # resetting interval losses for the next interval
                interval_losses = []

                # evaluating on validation set
                val_loss, bleu_score = evaluate(model, val_dataloader, device, tokenizer, bleu_metric, 20)
                if master_process: # print only for process with local rank 1
                    print("-----------------------------------------------------------")
                    print(f"Validation Loss after {global_step} steps: {val_loss}")
                    print(f"Validation BLEU Score after {global_step} steps: {bleu_score}")
                    print("-----------------------------------------------------------")

                    metrics = {
                        "global_step": global_step,
                        "train_loss": average_loss,
                        "val_loss": val_loss,
                        "val_bleu_score": bleu_score
                    }
                    all_metrics.append(metrics)

                    with open("training_metrics.json", "w") as f:
                        json.dump(all_metrics, f)

                    # saving the model checkpoint if validation loss improved
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                        # saving the new model checkpoint
                        checkpoint_name = f"checkpoint_step_{global_step}"
                        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                        os.makedirs(checkpoint_path, exist_ok=True)  # creating directory if it doesn't exist
                        
                        model.module.save_pretrained(checkpoint_path)
                        tokenizer.save_pretrained(checkpoint_path)

                        # updating the best checkpoint step for folder name
                        best_checkpoint_step = global_step

                        # TODO: doesn't work properly - only deletes a few iterations ago, not the previous one
                        if best_checkpoint_step is not None:
                            for filename in os.listdir(checkpoint_dir):
                                if filename.startswith(f"checkpoint_step_") and filename != f"checkpoint_step_{best_checkpoint_step}.pt":
                                    try:
                                        step_number = int(filename.split("_")[-1])
                                        if step_number < (best_checkpoint_step - 200):
                                            previous_checkpoint_path = os.path.join(checkpoint_dir, filename)
                                            if os.path.isdir(previous_checkpoint_path):
                                                import shutil
                                                shutil.rmtree(previous_checkpoint_path)
                                    except ValueError:
                                        continue
        torch.cuda.synchronize()
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        if master_process:
            print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")

    return train_losses

# evaluation loop
def evaluate(model, val_dataloader, device, tokenizer, bleu_metric, max_batches=None):
    model.eval()
    val_losses = []
    bleus = []
    num_evaluated_batches = 0

    with torch.no_grad():
        effective_batch_size = Config.batch_size_val * ddp_world_size
        max_steps = len(val_dataset) // effective_batch_size # max over the whole eval, but we use only 20 batches (steps)

        if max_batches is None:
            eval_iterator = tqdm(val_dataloader, desc=f"Evaluation", disable=ddp_local_rank != 0)
        else:
            eval_iterator = tqdm(val_dataloader, desc=f"Evaluation", total=max_batches, disable=ddp_local_rank != 0)

        for batch_idx, batch in enumerate(eval_iterator):
            if max_batches is not None and num_evaluated_batches >= max_batches:
                break
            
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            step_loss_tensor = loss.clone().detach().to(device)
            dist.all_reduce(step_loss_tensor, op=dist.ReduceOp.AVG)
            val_losses.append(step_loss_tensor.item())

            # generating predictions 
            generated_ids = model.module.generate(pixel_values, num_beams=4, max_length=256, early_stopping=True)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # computing BLEU scores
            bleu = bleu_metric.compute(predictions=generated_texts, references=label_texts)
            bleu = bleu['google_bleu']
            bleu_tensor = torch.tensor(bleu, device=device)
            dist.all_reduce(bleu_tensor, op=dist.ReduceOp.AVG)
            bleus.append(bleu_tensor.item())

            num_evaluated_batches += 1

    avg_val_loss = np.mean(val_losses)
    avg_bleu = np.mean(bleus)

    return avg_val_loss, avg_bleu

# starting LoRA fine-tuning
train_losses = train_lora(model, train_dataloader, optimizer, scheduler, device, num_epochs, eval_steps, val_dataloader, tokenizer, bleu_metric, local_rank=ddp_local_rank)

# loading the final evaluation dataset
eval_dataset = load_dataset("linxy/LaTeX_OCR", "human_handwrite")
combined_eval_ds = concatenate_datasets([eval_dataset['train'], eval_dataset['validation'], eval_dataset['test']])

if master_process:
    print("Test set size:", len(combined_eval_ds))
eval_dataset = LatexDataset(combined_eval_ds, tokenizer, feature_extractor, phase='test')

eval_dataloader = DataLoader(eval_dataset, batch_size=Config.batch_size_val, collate_fn=data_collator)

# evaluating on the final test dataset
checkpoint_dir = f"checkpoints/checkpoint_step_{best_checkpoint_step}"
best_model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir).to(device)

best_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

# evaluating on test set
test_loss, test_bleu_scores = evaluate(best_model, eval_dataloader, device, best_tokenizer, bleu_metric)
print(f"Test Loss: {test_loss}")
print(f"Test BLEU Score: {test_bleu_scores}")

if master_process: 
    metrics_test = {
            "test_losses": test_loss,
            "test_bleu_scores": test_bleu_scores
        }
    with open("test_metrics.json", "w") as f:
        json.dump(metrics_test, f)

dist.barrier()
dist.destroy_process_group()
