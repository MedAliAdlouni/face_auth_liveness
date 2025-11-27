import os
from facenet_pytorch import MTCNN

def detect_face(image):
    """Detect the (largest if more than one) face (if present) from the image.

    Arguments:
        img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

    Returns:
        tuple(torch.tensor, float) -- If detected, cropped image of a face
        with dimensions 3 x image_size x image_size. Optionally, the probability that a
        face was detected.
    """
    # face detection model instantiation
    mtcnn = MTCNN(image_size=160, margin=14, post_process=True)

    # face detection (forward pass through the model)
    face_tensor, detection_proba = mtcnn(image, return_prob=True)

    return face_tensor, detection_proba