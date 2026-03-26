import torch
from pathlib import Path
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader, random_split

class NEUClassificationDataset(Dataset):

    def __init__(self, images_dir: Path,
                annotations_dir: Path,
                transform=None):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform

        self.xml_files = list(self.annotations_dir.glob("*.xml"))

        self.classes = self._extract_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _extract_classes(self):
        classes = set()
        for xml_file in self.xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            label = root.find("object/name").text
            classes.add(label)
        return sorted(list(classes))
    
    def __len__(self):
        return len(self.xml_files)
    
    def __getitem__(self, idx):
        xml_path = self.xml_files[idx]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Image name
        filename = root.find("filename").text
        filename = Path(filename).stem + ".jpg"
        img_path = self.images_dir / filename

        # Label
        label_name = root.find("object/name").text
        label = self.class_to_idx[label_name]

        # Load image
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)
    
def build_dataset(images_dir: Path,
                   annotations_dir: Path,
                   transform):
    
    dataset = NEUClassificationDataset(images_dir=images_dir,
                                       annotations_dir=annotations_dir,
                                       transform=transform)
    return dataset

def build_splits(dataset,
                 train_ratio: float,
                 val_ratio: float,
                 seed: int = 42):
    
    torch.manual_seed(seed)

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    return random_split(dataset, [train_size, val_size, test_size])

def build_splits_index(dataset, train_ratio, val_ratio):

    dataset_size = len(dataset)

    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        range(dataset_size),
        [train_size, val_size, test_size]
    )

    return train_subset.indices, val_subset.indices, test_subset.indices

def build_dataloaders(train_dataset, 
                      val_dataset,
                      test_dataset,
                      batch_size: int,
                      num_workers: int,
                      pin_memory: bool):
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_dataloader, val_dataloader, test_dataloader
   
  