import torch
import pytest

# Now import the model
from rice_images.model import load_resnet18_timm

def test_resnet18_output_shape():
    # Load the model
    model = load_resnet18_timm(num_classes=5)
    model.eval()  # Set the model to evaluation mode

    # Define the input tensor with a shape of (1, 3, 224, 224) as expected for ResNet-18
    input_tensor = torch.randn(1, 3, 224, 224)

    # Perform a forward pass to get the output
    output = model(input_tensor)

    # Check that the output shape is (1, num_classes), where num_classes is 5
    assert output.shape == (1, 5), f"Expected output shape (1, 5), but got {output.shape}"

def test_model_output_type():
    model = load_resnet18_timm()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert isinstance(output, torch.Tensor), "Model output is not a tensor"

def test_model_invalid_input():
    model = load_resnet18_timm()

    # Test with invalid input (e.g., None)
    try:
        model(None)
    except Exception as e:
        print(f"Expected error for None input: {e}")

    # Test with an invalid input size (e.g., negative dimensions)
    try:
        dummy_input = torch.randn(-1, 3, 224, 224)
        model(dummy_input)
    except Exception as e:
        print(f"Expected error for invalid input size: {e}")



if __name__ == "__main__":
    assert True
