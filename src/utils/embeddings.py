import torch

def compare_embeddings(emb1, emb2, threshold=0.70):
    emb1 = emb1.squeeze()
    emb2 = emb2.squeeze()
    cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
    return cos_sim, cos_sim >= threshold