from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend import router as backend_router

# Create the main FastAPI app
app = FastAPI(
    title="Rice Image Classification API",
    description="API for classifying rice images using a pre-trained ResNet model.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Include routes from backend.py
app.include_router(backend_router)


@app.get("/health")
async def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    return {
        "status": "ok",
        "message": "API is running smoothly!",
    }
