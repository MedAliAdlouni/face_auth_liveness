import math
import random
import numpy as np
import logging

from src.detectors.face_detector import detect_face
from src.embedding.face_embedding import extract_face_embedding
from src.utils.embeddings import compare_embeddings
logger = logging.getLogger(__name__)


def verify_face(stored_embedding, captured_frames, similarity_threshold=0.65, max_samples=10):
    logger.info("\nPerforming face verification...")

    if not captured_frames:
        logger.error('No frames captured during liveness detection')
        return {
            'success': False,
            'message': 'No frames captured during liveness detection',
            'overall_passed': False
        }

    sample_count = min(max_samples, len(captured_frames))
    sampled_frames = random.sample(captured_frames, sample_count)
    logger.info(f"Analyzing {len(sampled_frames)} captured frames...\n")

    similarities = []
    for frame in sampled_frames:
        face_tensor, detection_prob = detect_face(frame)
        if face_tensor is None:
            continue

        live_embedding = extract_face_embedding(face_tensor)
        similarity, is_match = compare_embeddings(stored_embedding, live_embedding, threshold=similarity_threshold)
        similarities.append((float(similarity), bool(is_match)))

    if not similarities:
        logger.error('No faces detected in sampled frames for verification')
        return {
            'success': False,
            'message': 'No faces detected in sampled frames for verification',
            'overall_passed': False
        }

    matches = sum(1 for _, flag in similarities if flag)
    required = math.ceil(len(similarities) / 2)
    if matches < required:
        logger.warning(f"number of faces verified ({matches}) < required ({required})")
        return {
            'success': False,
            'message': 'Identity verification FAILED',
            'overall_passed': False
        }

    avg_similarity = float(np.mean([s for s, _ in similarities]))
    return {
        'success': True,
        'matches': matches,
        'evaluated_frames': len(similarities),
        'avg_similarity': avg_similarity
    }
