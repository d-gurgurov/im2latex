# Image-to-LaTeX Converter for Mathematical Formulas and Text

## Introduction

This project aims to train an encoder-decoder model that produces LaTeX code from images of formulas and text. Utilizing a diverse collection of image-to-LaTeX data, we compare the BLEU performance of our model with other similar models, such as [pix2tex](https://github.com/lukas-blecher/LaTeX-OCR) and [TexTeller](https://github.com/OleehyO/TexTeller/tree/main).

## Model Architecture

Our model leverages the architecture proposed in the [TrOCR](https://arxiv.org/abs/2109.10282) model by combining the Swin Transformer for image understanding and GPT-2 for text generation. 
