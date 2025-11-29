"""Model initialization utilities.

Provides explicit model warm-up and caching for startup.
"""
import logging

logger = logging.getLogger(__name__)


def init_models():
    """Pre-load and cache the face detection and embedding models.
    
    This should be called once at application startup (e.g., server boot)
    so that the expensive model initialization cost is paid upfront and
    hidden from user requests.
    
    Example usage:
        from src.models import init_models
        init_models()  # Call once in main() or FastAPI startup
    """
    logger.info("Initializing models...")
    
    # Import and call the lazy initializers
    from src.detectors.face_detector import get_mtcnn
    from src.embedding.face_embedding import get_resnet
    
    # First call initializes and caches the model
    logger.info("Loading MTCNN face detector...")
    get_mtcnn()
    logger.info("✓ MTCNN loaded")
    
    logger.info("Loading InceptionResnetV1 embedding model...")
    get_resnet()
    logger.info("✓ InceptionResnetV1 loaded")
    
    logger.info("Models initialized successfully!")
