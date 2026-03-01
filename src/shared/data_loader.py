import os
from pathlib import Path
import torch
from torchvision import datasets, transforms


# Data directory - always use current working directory (project root when using uv run)
DATA_DIR = Path.cwd() / "data"

def get_mnist(split="train", data_dir=None):
    """
    Load standard MNIST dataset.
    
    Args:
        split: "train" or "test"
        data_dir: directory to save data (defaults to project root/data)
    
    Returns:
        MNIST dataset with standard normalization
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir).resolve()
    
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(
        root=str(data_dir),
        train=(split == "train"),
        download=True,
        transform=transform
    )
    
    return dataset


def get_binarized_mnist(split="train", data_dir=None):
    """
    Load MNIST dataset with binarization (threshold at 0.5).
    Values > 0.5 become 1, else 0.
    
    Args:
        split: "train" or "test"
        data_dir: directory to save data (defaults to project root/data)
    
    Returns:
        MNIST dataset with binarized images
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir).resolve()
    
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    
    dataset = datasets.MNIST(
        root=str(data_dir),
        train=(split == "train"),
        download=True,
        transform=transform
    )
    
    return dataset


def get_mnist_raw(split="train", data_dir=None):
    """
    Load MNIST dataset as raw tensor [0, 1] without any normalization.
    Suitable for diffusion models that apply custom preprocessing.
    
    Args:
        split: "train" or "test"
        data_dir: directory to save data (defaults to project root/data)
    
    Returns:
        MNIST dataset with raw pixel values as tensors
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir).resolve()
    
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset = datasets.MNIST(
        root=str(data_dir),
        train=(split == "train"),
        download=True,
        transform=transform
    )
    
    return dataset