import argparse
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import wandb
import random
import numpy as np
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification Training Script')
    
    # Required arguments
    parser.add_argument('-d', '--datapath', type=str, required=True,
                       help='Path to the dataset directory containing train/val folders')
    
    # Training parameters
    parser.add_argument('-e', '--epochs', type=int, default=5,
                       help='Number of training epochs (default: 5)')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('-lr', '--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                       choices=['sgd', 'adam', 'nadam', 'adagrad', 'rmsprop'],
                       help='Optimizer to use (default: adam)')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer (default: 0.9)')
    
    # Model parameters
    parser.add_argument('-s', '--strategy', type=str, default='start',
                       choices=['start', 'middle', 'end', 'freeze_all'],
                       help='Layer freezing strategy (default: start)')
    parser.add_argument('-k', '--k', type=int, default=5,
                       help='Number of layers to freeze (default: 5)')
    
    # Data parameters
    parser.add_argument('-i', '--image_size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('-a', '--augmentation', type=str, default='No',
                       choices=['Yes', 'No'],
                       help='Use data augmentation (default: No)')
    
    # Logging and misc
    parser.add_argument('-l', '--log', type=int, default=0, choices=[0, 1],
                       help='Enable wandb logging (default: 0)')
    parser.add_argument('-dl', '--detailed_logs', type=int, default=1, choices=[0, 1],
                       help='Print detailed logs (default: 1)')
    parser.add_argument('-sd', '--seed', type=int, default=2,
                       help='Random seed (default: 2)')
    
    return parser.parse_args()

def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)
    
def load_data(batch_size, img_size, augmentation="No", device="cpu", detailed_logs=1, data_path=None):
    base_transforms = [transforms.Resize((img_size, img_size))]

    if augmentation == "Yes":
        extra_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30)
        ]
        base_transforms.extend(extra_transforms)

    training_pipeline = transforms.Compose([
        *base_transforms,
        transforms.ToTensor()
    ])

    evaluation_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    training_set = datasets.ImageFolder(
        root=os.path.join(data_path, 'train'),
        transform=training_pipeline
    )
    testing_set = datasets.ImageFolder(
        root=os.path.join(data_path, 'val'),
        transform=evaluation_pipeline
    )

    total_train = len(training_set)
    train_subset_size = int(0.8 * total_train)
    validation_subset_size = total_train - train_subset_size

    train_split, valid_split = random_split(
        training_set,
        [train_subset_size, validation_subset_size]
    )

    loader_config = {
        'batch_size': batch_size,
        'shuffle': True
    }

    if device == "cuda":
        loader_config['num_workers'] = 4

    if detailed_logs:
        print("Dataset processing completed successfully")

    return (
        training_set.classes,
        DataLoader(train_split, **loader_config),
        DataLoader(valid_split, **loader_config),
        DataLoader(testing_set, **loader_config)
    )
    
def freeze_layers(model, mode, k):
    layer_list = list(model.named_children())
    total = len(layer_list)

    if k < 0 or k >= total:
        raise ValueError(f"k must be between 0 and {total - 1}")

    if mode == "start":
        for i, (_, layer) in enumerate(layer_list, 1):
            if i <= k:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"Frozen first {k} layers")

    elif mode == "middle":
        mid = total // 2
        for i, (_, layer) in enumerate(layer_list, 1):
            if mid - k <= i < mid + k:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"Frozen layers from {mid - k} to {mid + k}")

    elif mode == "end":
        start = total - k
        for i, (_, layer) in enumerate(layer_list, 1):
            if i >= start:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"Frozen last {k} layers")

    elif mode == "freeze_all":
        for i, (_, layer) in enumerate(layer_list):
            if i < total - 1:
                for param in layer.parameters():
                    param.requires_grad = False
        print("All layers frozen except the last one")
 
def set_optimizer(model, optimizer_type, params):
    learning_rate = params["eta"]
    weight_decay = 0

    if optimizer_type == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=params["momentum"], weight_decay=weight_decay)
    elif optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "nadam":
        return optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "adagrad":
        return optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def train_model(model, device, PARAM, log, return_model=0):
    # Freeze specified layers based on strategy
    freeze_layers(model, PARAM["strategy"], PARAM["k"])

    # Initialize wandb logging if enabled
    if log == 1:
        wandb.init(project='DA6401_A2')
        wandb.run.name = 'PARTB'

    # Set up loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = set_optimizer(model, PARAM["optimizer"], PARAM)

    # Load and prepare data
    class_labels, train_loader, val_loader, test_loader = load_data(
        PARAM["batch_size"], 
        PARAM["image_size"], 
        PARAM["augmentation"], 
        device,
        PARAM["detailed_logs"],
        PARAM["data_path"]
    )

    # Training loop
    for epoch in range(PARAM["epochs"]):
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        total_samples = 0

        # Process training batches
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_images)
            batch_loss = loss_fn(predictions, batch_labels)
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            _, predicted_classes = torch.max(predictions.data, 1)
            total_samples += batch_labels.size(0)
            correct_train += (predicted_classes == batch_labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_images)
                v_loss = loss_fn(val_outputs, val_labels)
                val_loss += v_loss.item()
                _, val_pred = torch.max(val_outputs.data, 1)
                total_val += val_labels.size(0)
                correct_val += (val_pred == val_labels).sum().item()

        # Print epoch statistics
        print(f"Epoch {epoch+1}, "
              f"Training Loss: {epoch_loss/len(train_loader)}, "
              f"Training Accuracy: {100 * correct_train / total_samples}%, "
              f"Validation Loss: {val_loss/len(val_loader)}, "
              f"Validation Accuracy: {100 * correct_val / total_val}%")

        # Log metrics if enabled
        if log == 1:
            wandb.log({
                'Epochs': epoch+1,
                'Training Loss': epoch_loss/len(train_loader),
                'Training Accuracy': 100 * correct_train / total_samples,
                'Validation Loss': val_loss/len(val_loader),
                'Validation Accuracy': 100 * correct_val / total_val
            })

    # Finalize logging if enabled
    if log == 1:
        wandb.finish()

    # Return either model or validation accuracy
    return model if return_model == 1 else 100 * correct_val / total_val

def load_model(device):
    model = models.googlenet(pretrained=True)
    last_layer_in_features = model.fc.in_features
    model.fc = nn.Linear(last_layer_in_features, 10)
    model = model.to(device)
    return model

def load_test_data(img_size, device, data_path):
    test_augmentation = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    
    test_dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=test_augmentation)
    label = test_dataset.classes
    
    selected_samples = []
    for i in range(len(label)):
        random_numbers = random.sample(range(200), 3)
        for j in range(3):
            index = 200 * i + random_numbers[j]
            selected_samples.append(test_dataset.samples[index])

    selected_dataset = datasets.ImageFolder(data_path, transform=test_augmentation)
    selected_dataset.samples = selected_samples
    if device == "cuda":
        test_loader = DataLoader(selected_dataset, batch_size=1, num_workers=2, shuffle=True)
    else:
        test_loader = DataLoader(selected_dataset, batch_size=1, shuffle=True)

    return label, test_loader

def calculate_accuracy_on_test_data(model, device, data_path):
    labels, train_loader, val_loader, test_loader = load_data(
        1, 224, "No", device, 1, data_path
    )
    model.eval()
    running_test_loss = 0.0
    correct_pred = 0
    total_pred = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for test_img, test_label in test_loader:
            test_img = test_img.to(device)
            test_label = test_label.to(device)
            test_output = model(test_img)
            loss_test = criterion(test_output, test_label)
            running_test_loss += loss_test.item()
            _, class_ = torch.max(test_output.data, 1)
            total_pred += test_label.size(0)
            correct_pred += (class_ == test_label).sum().item()
    
    print(f"Test Accuracy : {100 * correct_pred / total_pred}, Test Loss : {running_test_loss/len(test_loader)}")


args = parse_args()

# Set random seeds
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

PARAM = {
    "data_path": args.datapath,
    "image_size": args.image_size,
    "batch_size": args.batch_size,
    "eta": args.lr,
    "epochs": args.epochs,
    "output_size": 10,
    "optimizer": args.optimizer,
    "momentum": args.momentum,
    "weight_decay": 0,
    "strategy": args.strategy,
    "k": args.k,
    "augmentation": args.augmentation,
    "detailed_logs": args.detailed_logs
}

device = set_device()
model = load_model(device)
tuned_model = train_model(model, device, PARAM, args.log, 1)
calculate_accuracy_on_test_data(tuned_model, device, args.datapath)
