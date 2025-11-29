import os
import torch
from facenet_pytorch import MTCNN

_mtcnn = None

def get_mtcnn(device: str | None = None):
    """Lazily create and return a single MTCNN instance.

    Keeping a single module-level instance avoids re-loading model weights and
    repeated CUDA allocations which are slow and can cause OOM.
    """
    global _mtcnn
    if _mtcnn is None:
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        _mtcnn = MTCNN(image_size=160, margin=14, post_process=True, device=device)
    return _mtcnn


def detect_face(image):
    """Detect (and crop) the largest face from `image`.

    Returns:
        tuple(torch.Tensor|None, float|None): cropped face tensor and detection probability.
    """
    mtcnn = get_mtcnn()
    face_tensor, detection_proba = mtcnn(image, return_prob=True)
    return face_tensor, detection_proba