"""Configuration constants for the project.

Centralize thresholds, paths, and device choices here so they are easy to
inspect and change during experiments and tests.
"""
import os
import torch

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EMBEDDINGS_DIR = os.environ.get('EMBEDDINGS_DIR', os.path.join(BASE_DIR, 'data', 'embeddings'))
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Device selection
DEFAULT_DEVICE = os.environ.get('DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')

# Liveness / embedding thresholds
DEFAULT_SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', 0.55))
DEFAULT_LIVENESS_DURATION = int(os.environ.get('LIVENESS_DURATION', 10))
TEXTURE_THRESHOLD = float(os.environ.get('TEXTURE_THRESHOLD', 25.0))

# Other defaults
MAX_FRAME_SAMPLES = int(os.environ.get('MAX_FRAME_SAMPLES', 20))
