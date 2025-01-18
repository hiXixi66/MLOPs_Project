import os
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image
from rice_images.model import load_resnet18_timm
from rice_images.data import load_data
from torchvision import transforms
import io
from contextlib import asynccontextmanager
import json
import anyio  # You need to install 'anyio' for async file handling

# Define the lifespan context manager before using it in FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model, transform

    # Define the relative path to the model from the current script location
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "models",
        "tester2",
        "resnet18_rice_final.pth",
    )

    # Load model
    # Ensure this matches your model architecture
    model = load_resnet18_timm(num_classes=5)
    model.load_state_dict(torch.load(model_path, weights_only=True))
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

    yield

    # Clean up
    del model
    del transform


# Now, instantiate FastAPI with the lifespan context manager
app = FastAPI(lifespan=lifespan)  # Use lifespan context manager with FastAPI


def predict_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    softmax_output = output.softmax(dim=-1)  # Apply softmax to the output
    _, predicted_idx = torch.max(output, 1)
    # Return both softmax output (probabilities) and predicted class index
    return softmax_output, predicted_idx.item()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}


@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        # Read the image file into memory (no need to save it to disk)
        contents = await file.read()

        # Log the size of the image and check if it's loaded correctly
        print(
            f"Received file {file.filename} with size {len(contents)} bytes."
        )

        # Save the file asynchronously using anyio
        async with await anyio.open_file(f"temp_{file.filename}", "wb") as f:
            # Ensure you await the write operation here
            await f.write(contents)

        # Predict the class for the image (pass the image bytes, not the filename)
        probabilities, predicted_class = predict_image(
            contents
        )  # Corrected: pass `contents` (image bytes)

        # Assuming your classes are named after the folder names: Arborio, Basmati, etc.
        class_names = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
        prediction = class_names[predicted_class]

        return {
            "filename": file.filename,
            "prediction": prediction,
            "probabilities": probabilities.tolist(),
        }

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        # Delete the temporary file after prediction
        if os.path.exists(f"temp_{file.filename}"):
            os.remove(f"temp_{file.filename}")
            print(f"Deleted temporary file: temp_{file.filename}")
