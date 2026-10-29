import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "data" / "processed" / "train"
VALID_DIR = BASE_DIR / "data" / "processed" / "valid"

IMG_SIZE = 224
BATCH_SIZE = 32
# num_workers=0 avoids Windows multiprocessing overhead; set higher on Linux/macOS
NUM_WORKERS = 0 if os.name == "nt" else 4

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),  # handles zoom variation
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,        # leaf colour robustness
                           saturation=0.3, hue=0.05),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])            # ImageNet stats
])

valid_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

def get_dataloaders():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(VALID_DIR, transform=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, valid_loader, train_dataset.classes

if __name__ == "__main__":
    train_loader, valid_loader, classes = get_dataloaders()
    print(f"Number of classes: {len(classes)}")
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")