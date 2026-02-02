### LIBRARIES ###

import os
import time
import copy
import random
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### SETUP LOGGING SYSTEM ###

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

### REPRODUCIBILITY ###

def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### DATALOADER ###

def get_dataset(name: str, use_cnn: bool = False) -> Tuple[DataLoader, DataLoader, int, int, int]:
    data_path = './data'
    if name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
        in_dim, img_size = 1, 28
    elif name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        in_dim, img_size = 3, 32
    else:
        loaders = {"Iris": load_iris, "Wine": load_wine}
        data = loaders[name]()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        return DataLoader(train_ds, batch_size=16, shuffle=True), DataLoader(test_ds, batch_size=16), data.data.shape[1], len(np.unique(data.target)), 0

    return DataLoader(train_set, batch_size=64, shuffle=True), DataLoader(test_set, batch_size=64), in_dim, len(train_set.classes), img_size

### QUANTIZATION ###

class FakeQuantize(nn.Module):
    """
    Simulates quantization error during training (QAT) or inference (PTQ).
    Improved version with activation tracking support.
    """
    def __init__(self, bits: int = 32, enabled: bool = True):
        super().__init__()
        self.bits = bits
        self.enabled = enabled
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.bits >= 32:
            return x
        
        if self.bits == 1:
            scale = x.abs().mean()
            # Straight-Through Estimator (STE)
            return (torch.sign(x) * scale).detach() + (x - x.detach())
        
        q_max = (2**(self.bits - 1)) - 1
        q_min = -(2**(self.bits - 1))
        
        # Symmetric quantization
        alpha = x.abs().max()
        scale = alpha / q_max if alpha > 0 else 1.0
        
        # Fake Quantization with STE
        q_x = torch.round(x / scale).clamp(q_min, q_max)
        dq_x = q_x * scale
        return dq_x + (x - x.detach())

class QuantizedLinear(nn.Linear):
    """Linear layer with support for both PTQ and QAT."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, bits: int = 32):
        super().__init__(in_features, out_features, bias)
        self.bits = bits
        self.weight_quantizer = FakeQuantize(bits)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.weight_quantizer(self.weight)
        return nn.functional.linear(x, q_weight, self.bias)

class QuantizedConv2d(nn.Conv2d):
    """2D Convolutional layer with support for both PTQ and QAT."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bits=32):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bits = bits
        self.weight_quantizer = FakeQuantize(bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.weight_quantizer(self.weight)
        return nn.functional.conv2d(x, q_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

### MODELS ###

class QuantizedCNN(nn.Module):
    """A Convolutional Neural Network with full quantization support."""
    def __init__(self, in_channels: int, out_features: int, input_size: int = 28, bits: int = 32):
        super().__init__()
        self.bits = bits
        self.activation_quantizer = FakeQuantize(bits)
        
        self.features = nn.Sequential(
            QuantizedConv2d(in_channels, 16, kernel_size=3, padding=1, bits=bits),
            nn.ReLU(),
            nn.MaxPool2d(2),
            self.activation_quantizer,
            
            QuantizedConv2d(16, 32, kernel_size=3, padding=1, bits=bits),
            nn.ReLU(),
            nn.MaxPool2d(2),
            self.activation_quantizer
        )
        
        feature_size = input_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            QuantizedLinear(32 * feature_size * feature_size, 128, bits=bits),
            nn.ReLU(),
            self.activation_quantizer,
            QuantizedLinear(128, out_features, bits=bits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input quantization
        x = self.activation_quantizer(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def set_quantization(self, bits: int, enabled: bool = True):
        """Enable/Disable quantization and set bit-width for all layers."""
        self.bits = bits
        self.activation_quantizer.bits = bits
        self.activation_quantizer.enabled = enabled
        for m in self.modules():
            if isinstance(m, (QuantizedLinear, QuantizedConv2d)):
                m.bits = bits
                m.weight_quantizer.bits = bits
                m.weight_quantizer.enabled = enabled

class QuantizedMLP(nn.Module):
    """A Multi-Layer Perceptron with full quantization support."""
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, bits: int = 32):
        super().__init__()
        self.bits = bits
        self.activation_quantizer = FakeQuantize(bits)
        
        self.layers = nn.Sequential(
            QuantizedLinear(in_features, hidden_dim, bits=bits),
            nn.ReLU(),
            self.activation_quantizer,
            QuantizedLinear(hidden_dim, out_features, bits=bits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_quantizer(x)
        return self.layers(x)

    def set_quantization(self, bits: int, enabled: bool = True):
        self.bits = bits
        self.activation_quantizer.bits = bits
        self.activation_quantizer.enabled = enabled
        for m in self.modules():
            if isinstance(m, QuantizedLinear):
                m.bits = bits
                m.weight_quantizer.bits = bits
                m.weight_quantizer.enabled = enabled

### METRICS & TRAIN ###

def calculate_model_size(model: nn.Module, bits: int) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * bits) / (8 * 1024)

def evaluate_model(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            correct += (output.argmax(dim=1) == target).sum().item()
            total += target.size(0)
    return {"loss": total_loss / len(loader), "acc": 100. * correct / total}

def train_model(model: nn.Module, train_loader: DataLoader, epochs: int = 5) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

### EXPERIMENT ###

def run_experiment(ds_name: str):
    set_seed(42)
    use_cnn = ds_name in ["MNIST", "CIFAR10"]
    use_cnn = ds_name in ["MNIST", "CIFAR10"]
    train_loader, test_loader, in_dim, out_dim, img_size = get_dataset(ds_name, use_cnn=use_cnn)
    
    bit_options = [32, 8, 4, 2, 1]
    qat_results = []
    ptq_results = []

    # QAT
    logger.info(f"Starting QAT experiments for {ds_name}...")
    for bits in bit_options:
        if use_cnn:
            model = QuantizedCNN(in_channels=in_dim, out_features=out_dim, input_size=img_size, bits=bits)
            epochs = 3
        else:
            model = QuantizedMLP(in_features=in_dim, hidden_dim=64, out_features=out_dim, bits=bits)
            epochs = 10
            
        train_model(model, train_loader, epochs=epochs)
        metrics = evaluate_model(model, test_loader)
        size_kb = calculate_model_size(model, bits)
        
        bit_label = "FP32" if bits == 32 else f"INT{bits}"
        qat_results.append({
            "Dataset": ds_name, "Bits": bit_label, "Loss": metrics['loss'], 
            "Accuracy (%)": metrics['acc'], "Size (KB)": size_kb
        })

    # PTQ
    logger.info(f"Starting PTQ experiments for {ds_name}...")
    # Train a baseline FP32 model first
    if use_cnn:
        base_model = QuantizedCNN(in_channels=in_dim, out_features=out_dim, input_size=img_size, bits=32)
        epochs = 3
    else:
        base_model = QuantizedMLP(in_features=in_dim, hidden_dim=64, out_features=out_dim, bits=32)
        epochs = 10
        
    train_model(base_model, train_loader, epochs=epochs)
    
    for bits in bit_options:
        ptq_model = copy.deepcopy(base_model)
        ptq_model.set_quantization(bits=bits, enabled=True)
        
        metrics = evaluate_model(ptq_model, test_loader)
        size_kb = calculate_model_size(ptq_model, bits)
        
        bit_label = "FP32" if bits == 32 else f"INT{bits}"
        ptq_results.append({
            "Dataset": ds_name, "Bits": bit_label, "Loss": metrics['loss'], 
            "Accuracy (%)": metrics['acc'], "Size (KB)": size_kb
        })

    return qat_results, ptq_results

def format_report(results: List[Dict], title: str):
    df = pd.DataFrame(results)
    df['Loss'] = df['Loss'].map('{:.4f}'.format)
    df['Accuracy (%)'] = df['Accuracy (%)'].map('{:.4f}'.format)
    df['Size (KB)'] = df['Size (KB)'].map('{:.4f}'.format)
    print(f"\nReport ({title}):")
    print(df.to_string(index=False))

if __name__ == "__main__":
    ds_to_run = "CIFAR10"
    qat_res, ptq_res = run_experiment(ds_to_run)
    
    format_report(qat_res, "QAT")
    format_report(ptq_res, "PTQ")