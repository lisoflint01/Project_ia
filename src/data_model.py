import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torchvision.models import resnet18, ResNet18_Weights


class my_IA(nn.Module):

    # Neaural Network for classification
    def __init__(self, num_classes: int, dropout_p: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            # convoluzione 3 (RGB), 16 output (IMG), kernel size 3 (SCORRIMENTO), padding 1 (RIEMPIMENTO)
            nn.Conv2d(3, 16, 3, padding=1),

            # normalizzazione per stabilizzare e accelerare l'addestramento
            nn.BatchNorm2d(16),

            #introduce non linearità nel modello, regole più complesse
            nn.ReLU(),

            # dimezza la dimensione (altezza e larghezza) dell'immagine
            nn.MaxPool2d(2),


            # prende l'output del primo blocco e lo elabora con un altro blocco
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),


            # prende l'output del secondo blocco e lo elabora con un altro blocco
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2),
            

            # prende l'output del terzo blocco e lo elabora con un altro blocco
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),


            # prende l'output del quarto blocco e lo elabora con un altro blocco
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),


            # Media globale: 1 valore per canale
            nn.AdaptiveAvgPool2d((1, 1)),
            # Da tensore a vettore
            nn.Flatten(),
            # Dropout per regolarizzazione
            nn.Dropout(p=dropout_p),
            # Layer di classificazione finale
            nn.Linear(64, num_classes)
        )

    # Model, pick image and return class
    def forward(self, x):
        return self.net(x)


# Select device
def pick_device(mode: str = None) -> torch.device:
    if mode is None:
        mode = "auto"
    mode = mode.lower()
    
    cuda_available = torch.cuda.is_available()
    
    # Default
    if mode == "auto":
        if cuda_available: 
            device = torch.device("cuda")
        else: 
            device = torch.device("cpu")
    
    # CPU
    elif mode == "cpu":
        device = torch.device("cpu")
    
    # Cuda and error cuda
    elif mode == "cuda":
        if cuda_available: 
            device = torch.device("cuda")
        else: 
            print ("request cuda but not able, got standard device")
            device = torch.device("cpu")
    
    # Gestion error
    else:
        device = torch.device("cpu")

    print(device)
        
    return device

# Set seed for python, numpy and torch for reproducibility
def set_seed(seed: int) -> None:

    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed_all(seed)


# Image trasform for train
def get_train_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Image trasform for val
def get_val_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
 
# Prepare training and validation datasets
def prepare_dataset(train_dir: str, val_dir: str, batch_size: int, img_size: int, out_dir: str, device: torch.device = None): 
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    out_dir = Path(out_dir)
    
    # Check train_dir
    if not train_dir.exists() or not train_dir.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    
    # Check val_dir
    if not val_dir.exists() or not val_dir.is_dir():
        raise FileNotFoundError(f"Val directory not found: {val_dir}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_transforms = get_train_transform(img_size)
    val_transforms = get_val_transform(img_size)
    
    train_dataset = ImageFolder(root=str(train_dir), transform=train_transforms)
    val_dataset = ImageFolder(root=str(val_dir), transform=val_transforms)
    
    train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
    idx_to_class = {value: key for key, value in train_dataset.class_to_idx.items()}
    
    # Directory labels
    labels_path = out_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)

    classes = train_dataset.classes
    
    return train_loader, val_loader, classes

# Build
def build_model(num_classes, freeze_backbone=False, pretrained=True, dropout=0.5, model_name="resnet18"):

    # Select Model and weights
    if model_name == "my_IA":
        return my_IA(num_classes, dropout)
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None
    
    model = resnet18(weights=weights)
        
    in_features = model.fc.in_features
    
    model.fc = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(512, num_classes))

    # Freeze backbone (weight)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model
