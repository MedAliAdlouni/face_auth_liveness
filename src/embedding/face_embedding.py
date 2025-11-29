import torch
from facenet_pytorch import InceptionResnetV1

_resnet = None
_resnet_device = None

def get_resnet(device: str | None = None):
    """Lazily create and return a single InceptionResnetV1 instance and device.

    Returns (model, device_str)
    """
    global _resnet, _resnet_device
    if _resnet is None:
        _resnet_device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        _resnet = InceptionResnetV1(pretrained='vggface2').eval().to(_resnet_device)
    return _resnet, _resnet_device


def extract_face_embedding(face):
    """Extract embeddings/CNN features from the face crop.

    `face` is expected to be a torch.Tensor (C,H,W) or (1,C,H,W).
    Returns a CPU tensor containing the embedding.
    """
    resnet, device = get_resnet()
    with torch.no_grad():
        if face.dim() == 3:
            face = face.unsqueeze(0)   # [C,H,W] → [1,C,H,W]
        face = face.to(device)
        embedding = resnet(face).detach().cpu()
    return embedding