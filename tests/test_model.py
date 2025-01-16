import sys
import os
import pytest
import torch

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

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

if __name__ == "__main__":
    assert True
    pytest.main()
