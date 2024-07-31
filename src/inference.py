from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
from datasets import load_dataset
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np


# loading model, tokenizer, and feature extractor
model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("DGurgurov/im2latex")
feature_extractor = AutoFeatureExtractor.from_pretrained("DGurgurov/im2latex")

# loading the dataset
new_dataset = load_dataset("linxy/LaTeX_OCR", "human_handwrite")
test_ds = new_dataset['test']

# dataset loading class
class LatexDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        feature_extractor,
        phase,
        image_size=(224, 468),
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
test_dataset = LatexDataset(test_ds, tokenizer, feature_extractor, phase='val')
test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=data_collator)

# making inference on test images
inferences = []
for batch_idx, batch in enumerate(test_dataloader):
    pixel_values = batch['pixel_values'].to("cuda")
    labels = batch['labels'].to("cuda")
    generated_ids = model.generate(pixel_values)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    inferences.append(generated_texts[0])
    

# saving inferences to a text file
output_file = "inferences.txt"
with open(output_file, "w") as f:
    for inference in inferences:
        f.write(inference + "\n")

print(f"Inferences saved to {output_file}")