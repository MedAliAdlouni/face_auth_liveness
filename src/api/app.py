"""FastAPI application for face authentication and liveness detection.

Exposes endpoints for user registration and authentication with live face verification.
"""
import os
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io

from src.models import init_models
from src.pipeline.registration_pipeline import register_pipeline
from src.pipeline.authentication_pipeline import authentication_pipeline
from src.config import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_LIVENESS_DURATION
from src.utils.verification import verify_face
from src.db import load_user_embedding_db
import cv2
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Face Authentication API",
    description="API for face recognition with liveness detection",
    version="1.0.0"
)

# Enable CORS for demo/clients (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve demo static files
static_dir = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(static_dir, exist_ok=True)
app.mount('/static', StaticFiles(directory=static_dir), name='static')


# ============================================================================
# Startup event: Initialize models once when the app starts
# ============================================================================
@app.on_event("startup")
def startup_event():
    """Called once when the server starts.
    
    Pre-loads the expensive ML models so they're ready for requests.
    """
    logger.info("🚀 Starting Face Authentication API...")
    init_models()
    logger.info("✓ API ready to serve requests")


# ============================================================================
# Response models (Pydantic for automatic validation and docs)
# ============================================================================
class RegisterResponse(BaseModel):
    success: bool
    user: Optional[str] = None
    message: Optional[str] = None
    embedding_path: Optional[str] = None


class AuthenticateResponse(BaseModel):
    success: bool
    user: Optional[str] = None
    overall_passed: bool
    liveness_passed: Optional[bool] = None
    liveness_confidence: Optional[float] = None
    avg_similarity: Optional[float] = None
    message: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================
@app.post("/register", response_model=RegisterResponse)
async def register(
    first_name: str = Form(...),
    last_name: str = Form(...),
    image: UploadFile = File(...)
):
    """Register a new user with their face image.
    
    Args:
        first_name: User's first name
        last_name: User's last name
        image: Face image file (PNG, JPG, etc.)
    
    Returns:
        Registration result with embedding path
    """
    try:
        logger.info(f"Registration request for {first_name} {last_name}")
        
        # Read image from upload
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Call registration pipeline
        result = register_pipeline(first_name, last_name, img)
        
        if result['success']:
            logger.info(f"✓ Registration successful for {first_name} {last_name}")
            return RegisterResponse(
                success=True,
                user=result.get('user'),
                embedding_path=result.get('embedding_path')
            )
        else:
            logger.warning(f"✗ Registration failed: {result['message']}")
            return RegisterResponse(
                success=False,
                message=result.get('message')
            )
    
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/authenticate", response_model=AuthenticateResponse)
async def authenticate(
    first_name: str = Form(...),
    last_name: str = Form(...),
    similarity_threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD),
    liveness_duration: int = Form(DEFAULT_LIVENESS_DURATION)
):
    """Authenticate a user via webcam liveness detection.
    
    Note: This endpoint expects the client to capture frames via webcam
    and call the authentication pipeline. For a REST API, you would
    typically send base64-encoded frames or stream via WebSocket.
    
    Args:
        first_name: User's first name
        last_name: User's last name
        similarity_threshold: Cosine similarity threshold (0.0-1.0)
        liveness_duration: Liveness check duration in seconds
    
    Returns:
        Authentication result with liveness and similarity scores
    """
    try:
        logger.info(f"Authentication request for {first_name} {last_name}")
        
        # Call authentication pipeline (triggers webcam)
        result = authentication_pipeline(
            first_name,
            last_name,
            similarity_threshold=similarity_threshold,
            liveness_duration=liveness_duration
        )
        
        if result['success']:
            logger.info(f"✓ Authentication successful for {first_name} {last_name}")
            return AuthenticateResponse(
                success=True,
                user=result.get('user'),
                overall_passed=result.get('overall_passed', False),
                liveness_passed=result.get('liveness_passed'),
                liveness_confidence=result.get('liveness_confidence'),
                avg_similarity=result.get('Average Similarity')
            )
        else:
            logger.warning(f"✗ Authentication failed: {result['message']}")
            return AuthenticateResponse(
                success=False,
                overall_passed=False,
                message=result.get('message')
            )
    
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/authenticate_frames", response_model=AuthenticateResponse)
async def authenticate_frames(
    first_name: str = Form(...),
    last_name: str = Form(...),
    files: List[UploadFile] = File(...),
    similarity_threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD)
):
    """Authenticate using frames uploaded from the client (browser).

    Accepts multiple image files (JPEG/PNG) captured on the client and
    runs `verify_face()` using those frames. This avoids opening server webcam.
    """
    try:
        # Load stored embedding (try DB first, then fallback to file-based helper)
        stored_embedding = load_user_embedding_db(first_name, last_name)
        if stored_embedding is None:
            return AuthenticateResponse(success=False, overall_passed=False, message="User not found")

        frames = []
        for f in files:
            content = await f.read()
            img = Image.open(io.BytesIO(content)).convert('RGB')
            arr = np.array(img)
            # convert RGB -> BGR for consistency with OpenCV code
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            frames.append(bgr)

        # Run verification on uploaded frames
        verify_res = verify_face(stored_embedding, frames, similarity_threshold=similarity_threshold)

        if not verify_res.get('success'):
            return AuthenticateResponse(success=False, overall_passed=False, message=verify_res.get('message'))

        return AuthenticateResponse(
            success=True,
            user=f"{first_name} {last_name}",
            overall_passed=True,
            liveness_passed=True,
            liveness_confidence=None,
            avg_similarity=verify_res.get('avg_similarity')
        )

    except Exception as e:
        logger.error(f"authenticate_frames error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
