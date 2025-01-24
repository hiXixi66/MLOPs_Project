import os
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File
from PIL import Image
from model import load_resnet18_timm
from torchvision import transforms
import io
import anyio

router = APIRouter()

# Global variables for model and transform
model = None
transform = None


def initialize_model():
    """Initialize the model and transform."""
    global model, transform

    if (
        model is None or transform is None
    ):  # Initialize only if not already done
        # Define the relative path to the model
        model_path = os.path.join(
            os.path.dirname(__file__),
            "../../models/tester2/resnet18_rice_final.pth",
        )

        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load the model
        model = load_resnet18_timm(num_classes=5)
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        model.eval()

        # Define the transform
        transform = transforms.Compose(
            [
                transforms.Resize((250, 250)),  # Resize to 250x250
                transforms.ToTensor(),  # Convert image to Tensor
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize
            ]
        )


@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}


@router.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        # Step 1: Log that the endpoint was hit
        print("Starting classification process...")

        # Step 2: Ensure the model is initialized
        print("Initializing model...")
        initialize_model()

        # Step 3: Read the image file into memory
        print(f"Reading file: {file.filename}")
        contents = await file.read()
        print(f"File size: {len(contents)} bytes.")

        # Step 4: Predict the class for the image
        print("Running prediction...")
        probabilities, predicted_class = predict_image(contents)

        # Step 5: Map prediction to class names
        print(f"Mapping prediction index {predicted_class} to class name.")
        class_names = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
        prediction = class_names[predicted_class]

        # Step 6: Return the result
        print(f"Prediction successful: {prediction}")
        return {
            "filename": file.filename,
            "prediction": prediction,
            "probabilities": probabilities.tolist(),
        }

    except Exception as e:
        # Step 7: Log the error
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {str(e)}"
        )


def predict_image(image_bytes: bytes):
    """Predict the class of the image."""
    global model, transform

    # Load and preprocess the image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0)

    # Run the model
    with torch.no_grad():
        output = model(img)

    # Apply softmax and get the predicted class
    softmax_output = output.softmax(dim=-1)
    _, predicted_idx = torch.max(output, 1)
    return softmax_output, predicted_idx.item()
