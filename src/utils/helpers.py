import torch
import os
import pickle
import cv2
from PIL import Image
import time 
import numpy as np


from src.liveness.liveness_detection import LivenessDetector

# Configuration
EMBEDDINGS_DIR = "data/embeddings"  # Directory to store user embeddings
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


def get_embedding_path(first_name, last_name):
    """
    Generate the file path for storing/loading user embedding.
    
    Args:
        first_name: User's first name
        last_name: User's last name
        
    Returns:
        str: Path to the embedding file
    """
    filename = f"{first_name.lower()}_{last_name.lower()}.pkl"
    return os.path.join(EMBEDDINGS_DIR, filename)


def compare_embeddings(emb1, emb2, threshold=0.70):  # LOWERED from 0.85
    """
    Compute cosine similarity between two face embeddings.
    """
    emb1 = emb1.squeeze()
    emb2 = emb2.squeeze()

    cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
    return cos_sim, cos_sim >= threshold


def load_user_embedding(first_name, last_name):
    """
    Load a user's stored embedding from file.
    
    Args:
        first_name: User's first name
        last_name: User's last name
        
    Returns:
        torch.Tensor: User's face embedding or None if not found
    """
    embedding_path = get_embedding_path(first_name, last_name)
    
    if not os.path.exists(embedding_path):
        return None
    
    with open(embedding_path, 'rb') as f:
        user_data = pickle.load(f)
    
    return user_data['embedding']



def upload_image(*args):
    """
    Upload image either from webcam or from provided path.
    Args:
        *args: If provided, should contain the image path.
    Returns:
        PIL.Image: Loaded image
    """
    image_choice = input("Use provided image path? If not, take image with webcam (y/n): ").strip().lower()
    if image_choice == 'n':
        print("\nOpening webcam... (photo will be taken in 5 seconds)")
        cam = cv2.VideoCapture(0)
        for i in range(5, 0, -1):
            ret, frame = cam.read()
            frame_display = frame.copy()
            cv2.putText(frame_display, f"{i}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 4)
            cv2.imshow("Webcam", frame_display)
            cv2.waitKey(1000)

        # Capture final frame
        ret, frame = cam.read()
        cam.release()
        cv2.destroyAllWindows()

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print("Image captured!")
    else:        
        img = Image.open(input("Enter path to face image: ")).convert('RGB')
    
    return img



def liveness_detection(duration=10):
        # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")
    
    detector = LivenessDetector()
    
    print("Face Liveness Detection Started")    
    start_time = time.time()
    
    all_results = []
    captured_frames = []
    face_detected_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        elapsed = time.time() - start_time
        remaining = max(0, duration - elapsed)
        
        if remaining == 0:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Store frame
        captured_frames.append(frame.copy())
        
        # Detect faces
        faces = detector.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        if len(faces) > 0:
            face_detected_frames += 1
        
        for (x, y, w, h) in faces:
            is_real, metrics = detector.is_real_face(frame, (x, y, w, h))
            
            all_results.append({
                'is_real': is_real,
                'metrics': metrics
            })
            
            color = (0, 255, 0) if is_real else (0, 0, 255)
            label = "REAL" if is_real else "FAKE/SPOOF"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            y_offset = y + h + 25
            line_height = 18
            
            cv2.putText(frame, f"Texture: {metrics['texture_score']:.1f} {'✓' if metrics['texture_passed'] else '✗'}", 
                       (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, f"Motion: {metrics['motion_score']:.1f} {'✓' if metrics['motion_natural'] else '✗'}", 
                       (x, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, f"Screen: {'Yes ✗' if metrics['is_screen_like'] else 'No ✓'}", 
                       (x, y_offset + line_height*4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, f"Score: {metrics['total_score']}/{metrics['max_score']}", 
                       (x, y_offset + line_height*5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        countdown_text = f"{int(remaining)}"
        cv2.putText(frame, countdown_text, (frame.shape[1]//2 - 30, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4)
        
        cv2.putText(frame, "Our AGI is checking if you are real or not ... ", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Face Liveness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    return detector, all_results, captured_frames




def analyze_liveness_results(all_results, detector):
    """
    Perform liveness detection and capture frames for face verification.
    
    Args:
        duration: Duration of liveness detection in seconds
        
    Returns:
        tuple: (liveness_passed, captured_frames, liveness_confidence)
    """

    # Analyze liveness results
    liveness_passed = False
    liveness_confidence = 0.0
    
    if all_results:
        print("\n" + "="*60)
        print("LIVENESS DETECTION RESULT")
        print("="*60)
        
        real_count = sum(1 for r in all_results if r['is_real'])
        total_count = len(all_results)
        
        avg_texture = np.mean([r['metrics']['texture_score'] for r in all_results])
        avg_motion = np.mean([r['metrics']['motion_score'] for r in all_results])
        screen_detections = sum(1 for r in all_results if r['metrics']['is_screen_like'])
        
        texture_ok = avg_texture > detector.texture_threshold
        motion_ok = 2 < avg_motion < 80
        screen_ok = screen_detections < (total_count * 0.3)
        
        criteria_passed = sum([texture_ok, motion_ok, screen_ok])
        probability = (real_count / total_count) * 100
        
        final_verdict = "REAL" if criteria_passed >= 2 and probability >= 60 else "FAKE"
        liveness_passed = (final_verdict == "REAL")
        liveness_confidence = min(probability, 95) if liveness_passed else max(100 - probability, 60)
        
        print(f"\nVerdict: {final_verdict}")
        print(f"Confidence: {liveness_confidence:.1f}%\n")
        
        print(f"Texture: {avg_texture:.1f} (need >{detector.texture_threshold}) {'✓' if texture_ok else '✗'}")
        print(f"Motion: {avg_motion:.1f} (need 2-80) {'✓' if motion_ok else '✗'}")
        print(f"Screen-like: {screen_detections}/{total_count} frames {'✓' if screen_ok else '✗'}")
        
        print(f"\nReason: ", end="")
        if liveness_passed:
            reasons = []
            if texture_ok: reasons.append("natural texture")
            if motion_ok: reasons.append("natural motion")
            if screen_ok: reasons.append("no screen artifacts")
            print(", ".join(reasons))
        else:
            reasons = []
            if not texture_ok: reasons.append("flat texture")
            if not motion_ok: reasons.append("unnatural motion")
            if not screen_ok: reasons.append("screen detected")
            print(", ".join(reasons))
        
        print("="*60 + "\n")
    else:
        print("\n✗ No face detected during liveness check!")
    
    return liveness_passed, liveness_confidence