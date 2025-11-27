import cv2
import numpy as np
from collections import deque
import time

from src.liveness.liveness_detector import LivenessDetector

def detect_liveness(duration=10):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    detector = LivenessDetector()
    
    print("Face Liveness Detection Started")    
    # Timer settings
    start_time = time.time()
    
    # Store results for final analysis
    all_results = []
    face_detected_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate remaining time
        elapsed = time.time() - start_time
        remaining = max(0, duration - elapsed)
        
        # Check if time is up
        if remaining == 0:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        if len(faces) > 0:
            face_detected_frames += 1
        
        for (x, y, w, h) in faces:
            # Determine if real or fake
            is_real, metrics = detector.is_real_face(frame, (x, y, w, h))
            
            # Store results
            all_results.append({
                'is_real': is_real,
                'metrics': metrics
            })
            
            # Draw rectangle and label
            color = (0, 255, 0) if is_real else (0, 0, 255)
            label = "REAL" if is_real else "FAKE/SPOOF"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Display live metrics
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
        
        # Display countdown timer (large and prominent)
        countdown_text = f"{int(remaining)}"
        cv2.putText(frame, countdown_text, (frame.shape[1]//2 - 30, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4)
        
        # Display instructions
        cv2.putText(frame, "Smile, life is beautiful :) ", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Face Liveness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # Analyze results
    if all_results:
        print("\n" + "="*60)
        print("LIVENESS DETECTION RESULT")
        print("="*60)
        
        # Calculate statistics
        real_count = sum(1 for r in all_results if r['is_real'])
        total_count = len(all_results)
        
        avg_texture = np.mean([r['metrics']['texture_score'] for r in all_results])
        avg_motion = np.mean([r['metrics']['motion_score'] for r in all_results])
        avg_brightness = np.mean([r['metrics']['brightness_std'] for r in all_results])
        screen_detections = sum(1 for r in all_results if r['metrics']['is_screen_like'])
        
        # Evaluation criteria
        texture_ok = avg_texture > detector.texture_threshold
        motion_ok = 2 < avg_motion < 80
        screen_ok = screen_detections < (total_count * 0.3)
        
        criteria_passed = sum([texture_ok, motion_ok, screen_ok])
        probability = (real_count / total_count) * 100
        
        final_verdict = "REAL" if criteria_passed >= 2 and probability >= 60 else "FAKE"
        confidence = min(probability, 95) if final_verdict == "REAL" else max(100 - probability, 60)
        
        print(f"\nVerdict: {final_verdict}")
        print(f"Confidence: {confidence:.1f}%\n")
        
        print(f"Texture: {avg_texture:.1f} (need >{detector.texture_threshold}) {'✓' if texture_ok else '✗'}")
        print(f"Motion: {avg_motion:.1f} (need 2-80) {'✓' if motion_ok else '✗'}")
        print(f"Screen-like: {screen_detections}/{total_count} frames {'✓' if screen_ok else '✗'}")
        
        print(f"\nReason: ", end="")
        if final_verdict == "REAL":
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
        
        print("="*60)
    else:
        print("\n No face detected!")


if __name__ == "__main__":
    main()