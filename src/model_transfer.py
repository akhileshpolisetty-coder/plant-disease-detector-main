import torch
import torch.nn as nn
from torchvision import models

def get_resnet_model(num_classes=38, freeze_base=True):
    # Load pretrained ResNet18 (trained on ImageNet)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_base:
        # Freeze all pretrained layers so only the new classifier head trains
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer to match our 38 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

if __name__ == "__main__":
    model = get_resnet_model(num_classes=38)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # should be [1, 38]

    # Sanity check: confirm only the final layer is trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")