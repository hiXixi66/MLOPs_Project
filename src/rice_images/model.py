import timm
import torch
import torch.nn as nn

def load_resnet18_timm(num_classes=5):
    """
    Load a pre-trained ResNet-18 model using timm and modify the final layer for a custom dataset.

    Args:
        num_classes (int): The number of classes for the final classification layer.

    Returns:
        torch.nn.Module: The modified ResNet-18 model.
    """
    # Load pre-trained ResNet-18 model from timm
    model = timm.create_model('resnet18', pretrained=True)

    # Replace the final classification layer to match num_classes
    # Check the number of input features for the classifier
    num_features = model.get_classifier().in_features

    # Set the classifier to a new fully connected layer
    model.fc = nn.Linear(num_features, num_classes)

    return model


if __name__ == "__main__":
    # Load the model with the desired number of output classes
    model = load_resnet18_timm()

    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    model.eval()  # Set the model to evaluation mode
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
