# Synthetic Image Generator Project

## Overview
This repository provides a configurable generator for creating synthetic grayscale microstructure images and corresponding binary segmentation masks.

It was developed for experiments in precipitate segmentation under limited annotated data, and can also be used to study how specific synthetic image properties affect segmentation model training.

## Generated Outputs
Each generated sample contains:
- `img.png` — synthetic grayscale image
- `mask.png` — binary precipitate mask
- `metadata.json` — generation metadata

Image size and dataset size are configurable.

Generation is controlled through JSON configuration files. Randomness is handled by explicit seeds to support reproducible dataset generation.


## Image Generation Components

### Matrix
The background is generated as a grain-like structure based on a Voronoi tessellation, with configurable:
- number of grains
- grayscale intensity range
- border width
- border irregularity
- shading
- Perlin and Gaussian texture noise

### Precipitates
Precipitates are generated as small dark irregular objects with configurable:
- number
- size range
- size distribution
- intensity
- local shading

### Etch Pits
Etch pits are generated as recessed elliptical features with configurable:
- number
- size range
- intensity range
- shading

## Typical Use
1. Edit a configuration file
2. Generate a dataset
3. Train a segmentation model on the generated data
4. Evaluate on real images


### Example Usage
The generator logic is exposed via the `generate()` function in `main.py`.

## Requirements
Python 3.10 or 3.11

```bash
pip install -r requirements.txt