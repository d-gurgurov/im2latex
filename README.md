# Image-to-LaTeX Converter for Mathematical Formulas and Text - im2latex

## Introduction

This project aims to train an encoder-decoder model that produces LaTeX code from images of formulas and text. Utilizing a diverse collection of image-to-LaTeX data, we build two models - a base 240M params model, capable of translating computer-generated formulas into LaTeX, and LoRa adapter, trained for translating hand-written formulas into LaTeX. Our base-model is available of [HuggingFace](https://huggingface.co/DGurgurov/im2latex), and the LoRa adapter will be available soon!

The paper describing our approach and the conducted experiments is available on [arXiv](https://arxiv.org/abs/2408.04015).

## Base-Model Architecture

Our base-model leverages the architecture proposed in the [TrOCR](https://arxiv.org/abs/2109.10282) model by combining the Swin Transformer for image understanding and GPT-2 for text generation. We start training the model by initializing its weights with the weights of these pre-trained models. 

<div align="center">
<img src="https://github.com/d-gurgurov/im2latex/blob/main/assets/im2latex.png?raw=true" alt="architecture" width="700"/>
</div>

### Training Curves

The following is the graph containing train and validation losses and BLEU score on validation: 
<div align="center">
<img src="https://github.com/d-gurgurov/im2latex/blob/main/assets/plots.png?raw=true" alt="training curves" width="500"/>
</div>

### Training Data
            
The data is taken from [OleehyO/latex-formulas](https://huggingface.co/datasets/OleehyO/latex-formulas). The data was divided into 80:10:10 for train, val and test. The splits were made as follows:

```python
dataset = load_dataset(OleehyO/latex-formulas, cleaned_formulas)
train_val_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_ds = train_val_split["train"]
val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)
val_ds = val_test_split["train"]
test_ds = val_test_split["test"]
```                     

### Test Evaluation Metrics

The model was evaluated on a test set with the following results:
- **Test Loss**: 0.10
- **Test BLEU Score**: 0.67

### Usage

You can use the model directly with the `transformers` library (inference.py is available):

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

## Model with LoRa Adapter Architecture

We enhance our pre-trained base model by integrating LoRa adapters into specific layers and blocks, enabling efficient fine-tuning for converting human hand-written formulas into LaTeX.

### Training Curves

The following is the graph containing train and validation losses and BLEU score on validation: 

<div align="center">
<img src="https://github.com/d-gurgurov/im2latex/blob/main/assets/lora_handwritten.png?raw=true" alt="training curves" width="500"/>
</div>

### Training Data
            
The data is taken from [linxy/LaTeX_OCR](https://huggingface.co/datasets/linxy/LaTeX_OCR). The data was originally split into train, val and test. The code for loading the datasets:

```python
new_dataset = load_dataset("linxy/LaTeX_OCR", "human_handwrite")

train_ds = new_dataset['train']
val_ds = new_dataset['validation']
test_ds = new_dataset['test']
```                     

### Test Evaluation Metrics

The model was evaluated on a test set with the following results:
- **Test Loss**: 0.02
- **Test BLEU Score**: 0.67

**Citation:**
- If you use this work in your research, please cite our paper:

```bibtex
@misc{gurgurov2024imagetolatexconvertermathematicalformulas,
      title={Image-to-LaTeX Converter for Mathematical Formulas and Text}, 
      author={Daniil Gurgurov and Aleksey Morshnev},
      year={2024},
      eprint={2408.04015},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.04015}, 
}
```


## License
[MIT]

