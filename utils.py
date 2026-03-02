import torch
import torchvision
from pathlib import Path
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import yaml

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_config(path: Path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
