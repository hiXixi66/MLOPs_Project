import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader

def load_resnet18(num_classes=5):
    # Load pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace the final fully connected layer for your custom dataset
    # ResNet-18's fc layer input features are 512
    model.fc = nn.Linear(512, num_classes)
    return model


if __name__ == "__main__":
    model = load_resnet18()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 3, 224, 224)
    model.eval() 
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")