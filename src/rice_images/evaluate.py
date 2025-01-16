from __future__ import annotations
import matplotlib.pyplot as plt
import torch
from rice_images.data import load_data
from rice_images.model import load_resnet18_timm
from torch import nn
import sys
import timm

DATA_PATH = "data/raw"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = load_resnet18_timm().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    train_dataset, val_dataset, test_dataset = load_data()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")
    
def main():
    model_checkpoint = sys.argv[1]
    evaluate(model_checkpoint) 
    
if __name__ == "__main__":
    main()