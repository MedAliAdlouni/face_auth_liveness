import numpy as np
import pytest

from src.utils import verification as vf


def fake_detect_face(frame):
    # Return a fake tensor (could be numpy) and a high confidence
    return np.zeros((3, 160, 160), dtype=np.uint8), 0.98


def fake_extract_face_embedding(tensor):
    # Return a deterministic embedding based on sum of tensor shape
    return np.ones((512,), dtype='float32') * 0.5


def fake_compare_embeddings(stored, live, threshold=0.65):
    # Return a fixed similarity and boolean based on threshold
    sim = 0.8
    return sim, sim >= threshold


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
    # Patch detect_face, extract_face_embedding, compare_embeddings used in the verification module
    monkeypatch.setattr(vf, 'detect_face', fake_detect_face)
    monkeypatch.setattr(vf, 'extract_face_embedding', fake_extract_face_embedding)
    monkeypatch.setattr(vf, 'compare_embeddings', fake_compare_embeddings)
    yield


def test_verify_face_success():
    # Create stored embedding (numpy)
    stored = np.ones((512,), dtype='float32') * 0.5
    # Create fake BGR frames (list of numpy arrays). Content irrelevant due to patching
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(6)]

    res = vf.verify_face(stored, frames, similarity_threshold=0.65, max_samples=6)
    assert res.get('success') is True
    assert res.get('matches') >= 1
    assert 'avg_similarity' in res


def test_verify_face_no_frames():
    stored = np.ones((512,), dtype='float32') * 0.5
    res = vf.verify_face(stored, [], similarity_threshold=0.65)
    assert res['success'] is False
    assert 'No frames captured' in res['message'] or 'No faces detected' in res['message']
