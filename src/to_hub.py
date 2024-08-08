from huggingface_hub import HfApi, Repository, login
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer
)

login(token='*')

# initializing API and repository
api = HfApi()
model_name = "DGurgurov/im2latex"
api.create_repo(repo_id=model_name, exist_ok=True)

# cloning the repository
checkpoint_dir = f"checkpoints/checkpoint_epoch_6_step_19400"

# saving model and tokenizer
api.upload_folder(
    folder_path=checkpoint_dir,
    path_in_repo='',
    repo_id=model_name
)


# Copy README to the repository
readme = """
# Your Model Name

This model is a VisionEncoderDecoderModel fine-tuned on a dataset for generating LaTeX formulas from images.

## Model Details

- **Encoder**: Swin Transformer
- **Decoder**: GPT-2
- **Framework**: PyTorch
- **DDP (Distributed Data Parallel)**: Used for training
            
## Training Data
            
The data is taken from [OleehyO/latex-formulas](https://huggingface.co/datasets/OleehyO/latex-formulas). The data was divided into 80:10:10 for train, val and test. The splits were made as follows:

```python
dataset = load_dataset(OleehyO/latex-formulas, cleaned_formulas)
train_val_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_ds = train_val_split["train"]
val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)
val_ds = val_test_split["train"]
test_ds = val_test_split["test"]
```                     

## Evaluation Metrics

The model was evaluated on a test set with the following results:
- **Test Loss**: 0.10473818009443304
- **Test BLEU Score**: 0.6661951245257148

## Usage

You can use the model directly with the `transformers` library:

```python
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import torch
from PIL import Image

# Load model, tokenizer, and feature extractor
model = VisionEncoderDecoderModel.from_pretrained("your-username/your-model-name")
tokenizer = AutoTokenizer.from_pretrained("your-username/your-model-name")
feature_extractor = AutoFeatureExtractor.from_pretrained("your-username/your-model-name")

# Prepare an image
image = Image.open("path/to/your/image.png")
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

# Generate LaTeX formula
generated_ids = model.generate(pixel_values)
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print("Generated LaTeX formula:", generated_texts[0])
```

## Training Script
The training script for this model can be found in the following repository: [GitHub](https://github.com/d-gurgurov/im2latex)

License
[MIT]
"""

with open(f'README.md', 'w') as f:
    f.write(readme)


api.upload_file(
path_or_fileobj=f'README.md',
path_in_repo='README.md',
repo_id=model_name
)
