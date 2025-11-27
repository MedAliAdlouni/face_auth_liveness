import os
import torch
from facenet_pytorch import InceptionResnetV1

def extract_face_embedding(face):
    """Extract embeddings/CNN features from the face crop.

    Arguments:
        face {torch.tensor} -- torch.Tensor.

    Returns:
        tuple(torch.tensor, float) -- If detected, cropped image of a face
        with dimensions 3 x image_size x image_size. Optionally, the probability that a
        face was detected.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Instantiate the face embedding extractor
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Extract face embedding (forward pass through the model)
    if face.dim() == 3:
        face = face.unsqueeze(0)   # [C,H,W] → [1,C,H,W]
    embedding = resnet(face).detach().cpu()
    
    return embedding