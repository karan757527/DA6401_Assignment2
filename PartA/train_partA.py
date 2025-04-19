import argparse
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a CNN on inaturalist dataset.')
parser.add_argument('-wp', '--wandb_project', type=str, default='DL-Assignment2',
                    help='Project name for Weights & Biases tracking.')
parser.add_argument('-we', '--wandb_entity', type=str, default='cs24m021',
                    help='Entity name for Weights & Biases tracking.')
parser.add_argument('-d', '--datapath', type=str, default="/Users/karanagrawal/Desktop/Sem 2/DL/inaturalist_12K",
                    help='Path to the dataset directory.')
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='Number of training epochs.')
parser.add_argument('-b', '--batch_size', type=int, default=64,
                    help='Batch size for training.')
parser.add_argument('-org', '--filter_org', type=str, default='double',
                    choices=['same', 'double', 'half', 'alternating_list', 'd_alternating_list', 'desc', 'asc'],
                    help='Filter organization strategy.')
parser.add_argument('-f_s', '--filter_size', type=int, nargs='+', default=[3,3,3,3,3],
                    help='List of filter sizes for convolutional layers.')
parser.add_argument('-f_n', '--filter_num', type=int, default=32,
                    help='Number of filters in the first convolutional layer.')
parser.add_argument('-pfs', '--pool_filter_size', type=int, default=2,
                    help='Pooling filter size.')
parser.add_argument('-dp', '--dropout', type=float, default=0.3,
                    help='Dropout rate for dense layers.')
parser.add_argument('-aug', '--augmentation', type=str, default='No', choices=['Yes', 'No'],
                    help='Enable data augmentation.')
parser.add_argument('-norm', '--batch_norm', type=str, default='Yes', choices=['Yes', 'No'],
                    help='Enable batch normalization.')
parser.add_argument('-img', '--image_size', type=int, default=224,
                    help='Input image size.')
parser.add_argument('-c_p', '--conv_padding', type=int, default=1,
                    help='Padding for convolutional layers.')
parser.add_argument('-c_s', '--conv_stride', type=int, default=1,
                    help='Stride for convolutional layers.')
parser.add_argument('-p_p', '--pool_padding', type=int, default=0,
                    help='Padding for pooling layers.')
parser.add_argument('-p_s', '--pool_stride', type=int, default=2,
                    help='Stride for pooling layers.')
parser.add_argument('-o', '--optimizer', type=str, default='adam',
                    choices=['sgd', 'rmsprop', 'adam', 'nadam', 'adagrad'],
                    help='Optimizer algorithm.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                    help='Learning rate.')
parser.add_argument('-m', '--momentum', type=float, default=0.9,
                    help='Momentum for SGD and RMSprop.')
parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,
                    help='Weight decay (L2 penalty).')
parser.add_argument('-w_i', '--weight_init', type=str, default='random',
                    choices=['random', 'Xavier'],
                    help='Weight initialization method.')
parser.add_argument('-ndl', '--neurons_fc', type=int, default=256,
                    help='Number of neurons in the fully connected layer.')
parser.add_argument('-a', '--activation', type=str, default='ReLU',
                    choices=['ReLU', 'LeakyReLU', 'GELU', 'SiLU', 'Mish'],
                    help='Activation function for hidden layers.')
parser.add_argument('-p', '--console', type=int, default=1, choices=[0, 1],
                    help='Print training metrics to console.')
parser.add_argument('-wl', '--wandb_log', type=int, default=0, choices=[0, 1],
                    help='Enable Weights & Biases logging.')
parser.add_argument('-dl', '--detailed_log', type=int, default=0, choices=[0, 1],
                    help='Enable detailed logging during setup.')
parser.add_argument('-plt', '--plot_grid', type=int, default=0, choices=[0, 1],
                    help='Plot sample test images.')
parser.add_argument('-eval', '--evaluate', type=int, default=0, choices=[0, 1],
                    help='Evaluate model on test set after training.')
args = parser.parse_args()

# Initialize Weights & Biases
wandb.login(key="843913992a9025996973825be4ad46e4636d0610")
wandb.init(project=args.wandb_project)

# Set device and seed
def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)
device = set_device()
print("Using device:", device)

torch.manual_seed(2)
random.seed(2)
np.random.seed(2)

# Construct PARAM dictionary
con_layers = len(args.filter_size)
dense_output_list = [args.neurons_fc]
dense_layers = len(dense_output_list)
dropout_list = [0.0] * con_layers + [args.dropout] * dense_layers

PARAM = {
    "con_layers": con_layers,
    "dense_layers": dense_layers,
    "filter_size": args.filter_size,
    "output_activation": "softmax",
    "dense_output_list": dense_output_list,
    "filter_num": args.filter_num,
    "filter_org": args.filter_org,
    "activation": args.activation,
    "input_channel": 3,
    "padding": args.conv_padding,
    "stride": args.conv_stride,
    "pool_padding": args.pool_padding,
    "pool_stride": args.pool_stride,
    "pool_filter_size": args.pool_filter_size,
    "image_size": args.image_size,
    "eta": args.learning_rate,
    "dropout": dropout_list,
    "epochs": args.epochs,
    "augmentation": args.augmentation,
    "batch_norm": args.batch_norm,
    "init": args.weight_init,
    "optimizer_name": args.optimizer,
    "batch_size": args.batch_size,
    "detailed_logs": args.detailed_log,
    "momentum": args.momentum,
    "weight_decay": args.weight_decay
}

DATA_PATH = args.datapath

# Data loading
def load_data(batch_size, img_size, augmentation="No", device="cpu", detailed_logs=1):
    base_transforms = [transforms.Resize((img_size, img_size))]
    if augmentation == "Yes":
        base_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30)
        ])
    training_pipeline = transforms.Compose([*base_transforms, transforms.ToTensor()])
    evaluation_pipeline = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

    train_set = datasets.ImageFolder(os.path.join(DATA_PATH, 'train'), training_pipeline)
    test_set = datasets.ImageFolder(os.path.join(DATA_PATH, 'val'), evaluation_pipeline)

    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_split, val_split = random_split(train_set, [train_size, val_size])

    loader_args = {'batch_size': batch_size, 'shuffle': True}
    if device == "cuda":
        loader_args['num_workers'] = 4

    return (
        train_set.classes,
        DataLoader(train_split, **loader_args),
        DataLoader(val_split, **loader_args),
        DataLoader(test_set, **loader_args)
    )

# Model definition
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, PARAM):
        super().__init__()
        self.detailed_logs = PARAM.get("detailed_logs", 0)
        self.flatten = nn.Flatten()
        self.filter_org = PARAM["filter_org"]
        self.filter_num = PARAM["filter_num"]
        self.con_layers = PARAM["con_layers"]
        self.den_layers = PARAM["dense_layers"]
        self.input_channel = PARAM["input_channel"]
        self.filter_size_list = PARAM["filter_size"]
        self.padding = PARAM["padding"]
        self.stride = PARAM["stride"]
        self.pool_padding = PARAM["pool_padding"]
        self.pool_stride = PARAM["pool_stride"]
        self.dense_output_list = PARAM["dense_output_list"]
        self.image_size = PARAM["image_size"]
        self.pool_filter_size = PARAM["pool_filter_size"]
        self.dropout_list = PARAM["dropout"]
        self.batch_norm = PARAM["batch_norm"]
        self.initialize = PARAM["init"]
        self.activation = PARAM["activation"]
        self.filter_num_list = self.organize_filters()
        self.act = self.activation_fun(self.activation)
        self.output_act = self.activation_fun("softmax")
        self.layers = nn.ModuleList()
        self.create_con_layers()

    def organize_filters(self):
        strategies = {
            "same": lambda: [self.filter_num] * self.con_layers,
            "double": lambda: [self.filter_num * (2 ** i) for i in range(self.con_layers)],
            "half": lambda: [self.filter_num // (2 ** i) for i in range(self.con_layers)],
            "alternating_list": lambda: [self.filter_num if i%2 == 0 else self.filter_num*2 for i in range(self.con_layers)],
            "d_alternating_list": lambda: [self.filter_num if i%4 in {0,1} else self.filter_num*2 for i in range(self.con_layers)],
            "desc": lambda: [self.filter_num - i for i in range(self.con_layers)],
            "asc": lambda: [self.filter_num + i for i in range(self.con_layers)]
        }
        return strategies.get(self.filter_org, lambda: [])()

    def activation_fun(self, act):
        activations = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "GELU": nn.GELU(),
            "SiLU": nn.SiLU(),
            "Mish": nn.Mish()
        }
        return activations.get(act, nn.ReLU())

    def create_con_layers(self):
        current_channels = self.input_channel
        current_size = self.image_size
        computations = 0
        for i in range(self.con_layers):
            layer = []
            conv = nn.Conv2d(current_channels, self.filter_num_list[i], self.filter_size_list[i],
                            padding=self.padding, stride=self.stride, bias=False)
            if self.initialize == "Xavier":
                nn.init.xavier_uniform_(conv.weight)
            layer.append(conv)
            if self.batch_norm == "Yes":
                layer.append(nn.BatchNorm2d(self.filter_num_list[i]))
            layer.append(self.act)
            layer.append(nn.MaxPool2d(self.pool_filter_size, stride=self.pool_stride, padding=self.pool_padding))
            if self.dropout_list[i] > 0:
                layer.append(nn.Dropout(self.dropout_list[i]))
            self.layers.append(nn.Sequential(*layer))
            # Update image size and channels
            current_channels = self.filter_num_list[i]
        # Dense layers
        dense_input = current_channels * (current_size // (2 ** self.con_layers)) ** 2
        for i in range(self.den_layers):
            dense = nn.Linear(dense_input, self.dense_output_list[i])
            layer = [dense, self.act]
            if self.dropout_list[self.con_layers + i] > 0:
                layer.append(nn.Dropout(self.dropout_list[self.con_layers + i]))
            self.layers.append(nn.Sequential(*layer))
            dense_input = self.dense_output_list[i]
        # Output layer
        self.layers.append(nn.Sequential(nn.Linear(dense_input, 10), nn.Softmax(dim=1)))

    def forward(self, x):
        for layer in self.layers[:self.con_layers]:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.layers[self.con_layers:-1]:
            x = layer(x)
        return self.layers[-1](x)

    def set_optimizer(self, optimizer_name, lr, momentum, weight_decay):
        optimizers = {
            'sgd': optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
            'rmsprop': optim.RMSprop(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
            'adam': optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay),
            'nadam': optim.NAdam(self.parameters(), lr=lr, weight_decay=weight_decay),
            'adagrad': optim.Adagrad(self.parameters(), lr=lr, weight_decay=weight_decay)
        }
        self.optimizer = optimizers.get(optimizer_name, optim.Adam(self.parameters(), lr=lr))
        return self.optimizer

# Training loop with early stopping
def train_model(model, device, PARAM, console_log, wandb_log):
    criterion = nn.CrossEntropyLoss()
    optimizer = model.set_optimizer(PARAM["optimizer_name"], PARAM["eta"], PARAM["momentum"], PARAM["weight_decay"])
    classes, train_loader, val_loader, test_loader = load_data(PARAM["batch_size"], PARAM["image_size"],
                                                              PARAM["augmentation"], device, PARAM["detailed_logs"])
    best_val_acc = 0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(PARAM["epochs"]):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100 * correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total

        if console_log:
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Epoch {epoch+1}: Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        if wandb_log:
            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss/len(train_loader),
                'train_acc': train_acc,
                'val_loss': val_loss/len(val_loader),
                'val_acc': val_acc
            })

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return best_val_acc

device = set_device()
net = ConvolutionalNeuralNetwork(PARAM).to(device)
accuracy_val = train_model(net, device, PARAM, args.console, args.wandb_log)
print(accuracy_val)