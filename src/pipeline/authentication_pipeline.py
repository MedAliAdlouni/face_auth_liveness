import cv2
import numpy as np
from PIL import Image
import time
import traceback
traceback.print_exc()

from src.embedding.face_embedding import extract_face_embedding
from src.utils.helpers import compare_embeddings, load_user_embedding, liveness_detection, analyze_liveness_results


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
    
    # Step 1: Load stored embedding
    print("[1/3] Loading stored embedding...")
    stored_embedding = load_user_embedding(first_name, last_name)
    
    if stored_embedding is None:
        print(f" User '{first_name} {last_name}' not found in database!")
        print(f"   Please register first using register_pipeline()")
        return {
            'success': False,
            'message': f"User '{first_name} {last_name}' not registered",
            'overall_passed': False
        }
    
    print(f" Embedding loaded for {first_name} {last_name}")
    
    # Step 2: Perform liveness detection
    print(f"\n[2/3] Starting liveness detection ({liveness_duration}s)...")
    print("Please look at the camera and move naturally.\n")
    detector, all_results, captured_frames = liveness_detection(duration=10)
    
    # Detect liveness
    liveness_passed, liveness_confidence = analyze_liveness_results(all_results, detector)
    
    if not liveness_passed:
        print("\n Liveness check FAILED - possible spoof detected!")
        return {
            'success': True,
            'overall_passed': False,
            'liveness_passed': False,
            'liveness_confidence': liveness_confidence,
            'face_verified': False,
            'message': 'Liveness detection failed'
        }
    
    # Step 3: Verify face
    print("\n[3/3] Performing face verification...")
    print(f"Analyzing {len(captured_frames)} captured frames...\n")
    
    similarities = []
    for captured_frame in captured_frames:
        live_embedding = extract_face_embedding(captured_frame)
        similarity, is_match = compare_embeddings(stored_embedding, live_embedding, threshold=similarity_threshold)
        similarities.append([similarity, is_match])
    
    if similarities[:][1] < (len(similarities) // 2):
        print(f" User '{first_name} {last_name}' looks fake!")
        return {
            'success': False,
            'message': f"User '{first_name} {last_name}' looks not real (fake/spoof)",
            'overall_passed': False
        }
    avg_similarity = np.mean(similarities[:][0])
    
    # Print results
    print(f"\nLiveness Detection:  {'✓ PASSED' if liveness_passed else '✗ FAILED'}")
    print(f"  Confidence: {liveness_confidence:.1f}%")
    print(f"  Average Similarity: {avg_similarity}")
    print(f"  Similarity Threshold:  {similarity_threshold}")
            
    return {
        'success': True,
        'user': f"{first_name} {last_name}",
        'liveness_passed': liveness_passed,
        'liveness_confidence': liveness_confidence,
        'Average Similarity': avg_similarity,
        'threshold': similarity_threshold
    }
        


if __name__ == "__main__":
    print("Face authentication system")    
    first_name = input("Enter first name: ").strip()
    last_name = input("Enter last name: ").strip()
    
    result = authentication_pipeline(first_name, last_name)
    if result['success'] and result['overall_passed']:
        print(f"\n ACCESS GRANTED for {first_name} {last_name}")
    else:
        print(f"\n ACCESS DENIED")