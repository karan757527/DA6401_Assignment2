# DA6401 Assignment 2: CNNs from Scratch & Fine-Tuning Pretrained Models

This repository contains two parts of the assignment for **DA6401 - Deep Learning**, focused on building and evaluating convolutional neural networks (CNNs) for image classification using the iNaturalist dataset.


## Overview

- **Part A**: Build a CNN from scratch, tune its hyperparameters, and evaluate it.
- **Part B**: Load a pretrained model (GoogLeNet), fine-tune it, and evaluate performance.

Each part uses the [iNaturalist dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip).

## WandB Report

Visualizations, sweep insights, and evaluation results are available in this [Weights & Biases Report](https://api.wandb.ai/links/cs24m021-iit-madras/nvfd4a1c).

## Part A - Train from Scratch

This part trains a CNN from scratch using configurable layers, activation functions, filter sizes, and more.

### Usage

```bash
python train_partA.py --config config.json
```

### Arguments

- `--datapath`: Path to the dataset directory.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training.
- `--filter_org`: Filter organization strategy. Choices: `same`, `double`, `half`, etc.
- `--filter_size`: List of filter sizes for convolutional layers.
- `--filter_num`: Number of filters in the first conv layer.
- `--pool_filter_size`: Pooling kernel size.
- `--dropout`: Dropout rate for dense layers.
- `--augmentation`: Enable data augmentation (`Yes`/`No`).
- `--batch_norm`: Enable batch normalization (`Yes`/`No`).
- `--image_size`: Input image size.
- `--conv_padding`, `--conv_stride`: Convolutional layer padding and stride.
- `--pool_padding`, `--pool_stride`: Pooling layer padding and stride.
- `--optimizer`: Optimizer type. Choices: `adam`, `sgd`, etc.
- `--learning_rate`: Learning rate.
- `--momentum`, `--weight_decay`: For optimizer tuning.
- `--weight_init`: Weight initialization (`random`, `Xavier`).
- `--neurons_fc`: Number of neurons in dense layer.
- `--activation`: Activation function (`ReLU`, `LeakyReLU`, `GELU`, etc.).
- `--evaluate`: Evaluate model on test set after training.
- `--wandb_log`: Enable Weights & Biases logging.
- `--plot_grid`: Plot predictions on test images.

## Part B - Finetune a Pretrained Model

This part fine-tunes a pretrained GoogLeNet using transfer learning.

### Usage

```bash
python train_partB.py --datapath ./data --model googlenet --freeze final --epochs 5 --batch_size 64 --lr 1e-4
```

### Arguments

- `--datapath`: Path to dataset directory containing `train/val` folders.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training.
- `--lr`: Learning rate.
- `--optimizer`: Optimizer to use. Options: `adam`, `sgd`, `rmsprop`, etc.
- `--model`: Pretrained model to use (e.g., `googlenet`).
- `--freeze`: Strategy for layer freezing. Options include `final`, `first_k`, `middle`, `none`.

## Best Configuration (Part A)

```python
PARAMS = {
    "con_layers": 5,
    "dense_layers": 1,
    "filter_size": [3] * 5,
    "output_activation": "softmax",
    "dense_output_list": [256],
    "filter_num": 32,
    "activation": "ReLU",
    "filter_org": "double",
    "image_size": 224,
    "pool_filter_size": 2,
    "batch_size": 64,
    "eta": 1.0e-4,
    "dropout": [0, 0, 0, 0, 0, 0.3],
    "epochs": 10,
    "augmentation": "No",
    "batch_norm": "Yes",
    "init": "Random",
    "input_channel": 3,
    "padding": 1,
    "stride": 1,
    "pool_padding": 0,
    "pool_stride": 2,
    "optimizer_name": 'adam',
    "detailed_logs": 1
}
```


## Installation

```bash
pip install -r requirements.txt
```
