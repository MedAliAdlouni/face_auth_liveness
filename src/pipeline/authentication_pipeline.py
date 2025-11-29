import numpy as np
import random
import traceback
import logging
import math

traceback.print_exc()

from src.embedding.face_embedding import extract_face_embedding
from src.utils.embeddings import compare_embeddings
from src.db import load_user_embedding_db
from src.utils.liveness_utils import perform_liveness
from src.detectors.face_detector import detect_face
from src.utils.verification import verify_face
from src.config import MAX_FRAME_SAMPLES 

# Set up logger for this module
logging.basicConfig(
    level=logging.INFO,                       # Show INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def authentication_pipeline(first_name, last_name, similarity_threshold=0.65, liveness_duration=10):
    """
    authentication pipeline: Verify a live user against their stored embedding.
    
    Pipeline steps:
        1. Load stored embedding for the user
        2. Perform liveness detection via webcam
        3. Extract face embedding from live capture
        4. Compare live embedding with stored embedding
        5. Return verification result
    
    Args:
        first_name: User's first name
        last_name: User's last name
        similarity_threshold: Cosine similarity threshold (default: 0.70)
        liveness_duration: Duration of liveness detection in seconds (default: 10)
        
    Returns:
        dict: Complete verification results including:
            - overall_passed: Boolean for complete authentication
            - liveness_passed: Boolean for liveness check
            - face_verified: Boolean for face match
            - best_similarity: Float similarity score
    """

    print(f"\nVerifying: {first_name} {last_name}\n")
    
    # Step 1: Load stored embedding from DB
    stored_embedding = load_user_embedding_db(first_name, last_name)
    if stored_embedding is None:
        result = {
            'success': False,
            'message': f"User '{first_name} {last_name}' not registered",
            'overall_passed': False
        }
        print(result)
        return result
    # Step 2: Liveness detection
    liveness_passed, liveness_confidence, detector, all_results, captured_frames = perform_liveness(liveness_duration)

    if not liveness_passed:
        logger.warning("\n Liveness check FAILED - possible spoof detected!")
        return {
            'success': False,
            'overall_passed': False,
            'liveness_passed': False,
            'liveness_confidence': liveness_confidence,
            'face_verified': False,
            'message': 'Liveness detection failed'
        }
    
    
    # Step 3: Verify face using the shared verification utility
    verify_res = verify_face(stored_embedding, captured_frames, similarity_threshold=similarity_threshold, max_samples=MAX_FRAME_SAMPLES)
    if not verify_res.get('success'):
        logger.warning("Identity verification FAILED for %s %s", first_name, last_name)
        return {
            'success': False,
            'message': verify_res.get('message', 'Identity verification FAILED'),
            'overall_passed': False
        }

    avg_similarity = verify_res.get('avg_similarity', None)
    
    # If verification succeeded, build final response
    logger.info(f"\nLiveness Detection:  {'✓ PASSED' if liveness_passed else '✗ FAILED'}")
    logger.info(f"  Confidence: {liveness_confidence:.1f}%")
    logger.info(f"  Average Similarity: {avg_similarity}")
    logger.info(f"  Similarity Threshold:  {similarity_threshold}")
    logger.info(f"Average similarity : {avg_similarity}")
            
    final_result = {
        'success': True,
        'user': f"{first_name} {last_name}",
        'liveness_passed': liveness_passed,
        'liveness_confidence': liveness_confidence,
        'Average Similarity': avg_similarity,
        'threshold': similarity_threshold
    }
    return final_result  


if __name__ == "__main__":
    print("Face authentication system")    
    first_name = input("Enter first name: ").strip()
    last_name = input("Enter last name: ").strip()
    
    result = authentication_pipeline(first_name, last_name)
    if result['success']:
        print(f"\n ACCESS GRANTED for {first_name} {last_name}")
    else:
        print(f"\n ACCESS DENIED")