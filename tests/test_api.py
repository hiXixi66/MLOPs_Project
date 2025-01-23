import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torchvision import transforms
from src.rice_images.api import app  # Import your FastAPI app
from src.rice_images.model import load_resnet18_timm


# Initialize the test client
@pytest.fixture
def client():
    """
    Provides a test client for the FastAPI app.
    """
    with TestClient(app) as c:
        yield c


# Define transformations (must match those used during training)
transform = transforms.Compose([
    transforms.Resize((250, 250)),  # Resize to 250x250
    transforms.ToTensor(),          # Convert image to Tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean values used during training
        std=[0.229, 0.224, 0.225]    # Standard deviation values used during training
    ),
])


def test_model_with_preprocessed_image():
    """
    Test the model directly with a preprocessed image tensor.
    """
    # Load and preprocess the image
    image_path = "data/raw/Rice_Image_Dataset/Rice_Image_Dataset/1/1.jpg"
    image = Image.open(image_path).convert("RGB")

    # Define the preprocessing steps (must match training)
    transform = transforms.Compose([
        transforms.Resize((250, 250)),  # Resize to 250x250
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Mean values from training
            std=[0.229, 0.224, 0.225]    # Standard deviation from training
        ),
    ])
    preprocessed_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Load the model
    model = load_resnet18_timm(num_classes=5)  # Replace with your model loading function
    model.load_state_dict(torch.load("path_to_your_model_weights.pt"))
    model.eval()  # Set model to evaluation mode

    # Perform inference
    with torch.no_grad():
        output = model(preprocessed_image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()

    # Map the predicted index to class names
    class_names = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Class probabilities: {probabilities.numpy()}")

    # Assert conditions (example: checking the highest probability is reasonable)
    assert probabilities.max() > 0.5, "Model confidence is too low"

def test_classify_no_file(client):
    """
    Test the /classify/ endpoint with no file provided.
    """
    response = client.post("/classify/")
    assert response.status_code == 422  # Unprocessable Entity