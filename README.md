# Image-to-LaTeX Converter for Mathematical Formulas and Text - im2latex

## Introduction

This project aims to train an encoder-decoder model that produces LaTeX code from images of formulas and text. Utilizing a diverse collection of image-to-LaTeX data, we compare the BLEU performance of our model with other similar models, such as [pix2tex](https://github.com/lukas-blecher/LaTeX-OCR) and [TexTeller](https://github.com/OleehyO/TexTeller/tree/main). Our model is available of [HuggingFace](https://huggingface.co/DGurgurov/im2latex).

## Model Architecture

Our model leverages the architecture proposed in the [TrOCR](https://arxiv.org/abs/2109.10282) model by combining the Swin Transformer for image understanding and GPT-2 for text generation. 

## Training Curves

The following is the graph containing train and validation losses and BLEU score on validation: 

![training curves](https://github.com/d-gurgurov/im2latex/blob/main/assets/plots.png?raw=true)

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

## Test Evaluation Metrics

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

## License
[MIT]

