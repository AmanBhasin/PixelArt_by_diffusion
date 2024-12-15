# DDPM Model for Pixel Art Generation

## Introduction
Pixel art generation presents unique challenges due to its inherently low resolution, sharp edges, and simple, discrete color palettes. Most generative models, including diffusion models, are designed for high-resolution, continuous-tone images, and thus often struggle to replicate these specific traits of pixel art.<br>
    </tab>project seeks to investigate the application of diffusion models for generating high-quality pixel art under low-resolution
constraints. The primary aim is to adapt diffusion models to produce pixel art while maintaining critical stylistic features such as sharp edges, discrete color transitions, and minimal pixel blending.

## Source codes and Implementation
- colorQ

## Data Directory

`data_dir`: Directory containing the dataset.

## Unet Model

`Unet.py`: Implementation of Unet model class.

## Custom Dataset

`Custom_dataset.py`: Implementation of class to load dataset in standard torch format.


## Training Notebook

`training-notebook`: Notebook for building, compiling, and training the model.

## Requirements

## Requirements

- Python 3.8 or higher
- PyTorch 1.9.0
- torchvision 0.10.0
- numpy 1.21.0
- pandas 1.3.0
- matplotlib 3.4.2


