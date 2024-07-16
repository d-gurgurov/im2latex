# Image-to-LaTeX Converter for Mathematical Formulas and Text - im2latex

## Introduction

This project aims to train an encoder-decoder model that produces LaTeX code from images of formulas and text. Utilizing a diverse collection of image-to-LaTeX data, we compare the BLEU performance of our model with other similar models, such as [pix2tex](https://github.com/lukas-blecher/LaTeX-OCR) and [TexTeller](https://github.com/OleehyO/TexTeller/tree/main). Our model is available of [HuggingFace](https://huggingface.co/DGurgurov/im2latex).

## Model Architecture

Our model leverages the architecture proposed in the [TrOCR](https://arxiv.org/abs/2109.10282) model by combining the Swin Transformer for image understanding and GPT-2 for text generation. 

<img src="https://github.com/d-gurgurov/im2latex/blob/main/assets/im2latex.png?raw=true" alt="architecture" width="700"/>

## Training Curves

The following is the graph containing train and validation losses and BLEU score on validation: 

<img src="https://github.com/d-gurgurov/im2latex/blob/main/assets/plots.png?raw=true" alt="training curves" width="500"/>

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
- **Test Loss**: 0.10
- **Test BLEU Score**: 0.67

## Usage

You can use the model directly with the `transformers` library:

```python
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import torch
from PIL import Image

# load model, tokenizer, and feature extractor
model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex")
tokenizer = AutoTokenizer.from_pretrained("DGurgurov/im2latex")
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k") # using the original feature extractor for now

# prepare an image
image = Image.open("path/to/your/image.png")
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

# generate LaTeX formula
generated_ids = model.generate(pixel_values)
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print("Generated LaTeX formula:", generated_texts[0])
```

## License
[MIT]

