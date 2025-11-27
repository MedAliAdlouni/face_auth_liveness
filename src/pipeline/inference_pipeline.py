import cv2
import torch
import numpy as np
from PIL import Image

from src.detectors.face_detector import detect_face
from src.embedding.face_embedding import extract_face_embedding
from src.liveness.liveness_detection import detect_liveness


def compare_embeddings(emb1, emb2, threshold=0.70):  # LOWERED from 0.85
    """
    Compute cosine similarity between two face embeddings.
    Lowered threshold to 0.70 for more realistic matching.
    """
    emb1 = emb1.squeeze()
    emb2 = emb2.squeeze()

    cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
    return cos_sim, cos_sim >= threshold


def main(duration=10):
    print("=== FACE VERIFICATION + LIVENESS PIPELINE ===")
    print()

    # ---------------------------------------------------------
    # PART 1 — LOAD USER INPUT IMAGE AND EXTRACT EMBEDDING
    # ---------------------------------------------------------
    input_path = input("Enter path to reference face image: ").strip()

    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot open image: {e}")
        return

    print("[INFO] Detecting face in input image…")
    ref_face_tensor, detection_proba = detect_face(img)

    if ref_face_tensor is None:
        print("[ERROR] No face detected in the reference image.")
        return

    print(f"[INFO] Face detected (p={detection_proba:.2f}). Extracting embedding…")

    ref_embedding = extract_face_embedding(ref_face_tensor)
    print("[INFO] Reference embedding extracted.\n")

    # ---------------------------------------------------------
    # PART 2 — CAMERA ACTIVATION FOR LIVE LIVENESS + ID MATCHING
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(0)
    
    # RELAXED liveness detector settings
    liveness = detect_liveness()
    # liveness.texture_threshold = 15  # LOWERED from 25
    
    print("Webcam started.")
    print("Look at the camera and move your head slightly.\n")

    start_time = cv2.getTickCount()
    fps = cv2.getTickFrequency()

    # Stats collection
    live_results = []
    similarity_scores = []
    face_detected_frames = 0
    
    # Add warmup period for motion detection
    warmup_frames = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from camera.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Time tracking
        elapsed = (cv2.getTickCount() - start_time) / fps
        remaining = max(0, duration - elapsed)
        if remaining == 0:
            break

        # Face detection (OpenCV Haar)
        faces = liveness.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)  # Larger minimum
        )

        if len(faces) > 0:
            face_detected_frames += 1

        for (x, y, w, h) in faces:
            # Skip warmup period for liveness
            if face_detected_frames <= warmup_frames:
                cv2.putText(frame, "Initializing...", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                continue
            
            # Liveness analysis
            is_real, metrics = liveness.is_real_face(frame, (x, y, w, h))

            # Identity verification - use LARGER crop with padding
            pad = 20
            y1 = max(0, y - pad)
            y2 = min(frame.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(frame.shape[1], x + w + pad)
            
            face_crop = frame[y1:y2, x1:x2]
            
            # Ensure crop is valid
            if face_crop.size == 0 or face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                continue
                
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_crop_pil = Image.fromarray(face_crop_rgb)

            live_face_tensor, _ = detect_face(face_crop_pil)

            same_person = False
            sim = 0

            if live_face_tensor is not None:
                live_embedding = extract_face_embedding(live_face_tensor)
                sim, same_person = compare_embeddings(ref_embedding, live_embedding)
                similarity_scores.append(sim)
            else:
                # If detection fails, skip this frame rather than counting as mismatch
                continue

            # Store complete results
            live_results.append({
                "is_real": is_real,
                "same_person": same_person,
                "similarity": sim,
                "metrics": metrics
            })

            # Drawing
            color = (0, 255, 0) if is_real and same_person else (0, 0, 255)
            label = f"{'REAL' if is_real else 'FAKE'} | {'MATCH' if same_person else 'NO MATCH'}"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, f"Sim: {sim:.2f}",
                        (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            
            # Show individual test results
            y_offset = y + h + 45
            cv2.putText(frame, f"Texture: {metrics['texture_score']:.1f} {'✓' if metrics['texture_passed'] else '✗'}",
                       (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Motion: {metrics['motion_score']:.1f} {'✓' if metrics['motion_natural'] else '✗'}",
                       (x, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Timer
        cv2.putText(frame, str(int(remaining)), (frame.shape[1]//2 - 30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4)

        cv2.imshow("Verification + Liveness", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # PART 3 — FINAL EVALUATION
    # ---------------------------------------------------------
    print("\n" + "="*65)
    print("FINAL RESULT")
    print("="*65)

    if len(live_results) == 0:
        print("❌ No valid face detected in camera feed. Cannot verify identity.")
        print("Tips: Ensure good lighting, face the camera directly, and stay still.")
        return

    # Liveness summary
    liveness_real_count = sum(1 for r in live_results if r["is_real"])
    liveness_score = (liveness_real_count / len(live_results)) * 100

    # Identity summary
    matches = sum(1 for r in live_results if r["same_person"])
    identity_score = (matches / len(live_results)) * 100
    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
    max_similarity = max(similarity_scores) if similarity_scores else 0

    print(f"Frames analyzed: {len(live_results)}")
    print(f"Liveness score: {liveness_score:.1f}%")
    print(f"Identity match score: {identity_score:.1f}%")
    print(f"Average similarity: {avg_similarity:.3f}")
    print(f"Max similarity: {max_similarity:.3f}")

    # More lenient thresholds
    final_liveness = liveness_score >= 50  # LOWERED from 60
    final_identity = identity_score >= 50 or max_similarity >= 0.70  # Added max similarity check

    print("\nVerdict:")
    if final_liveness and final_identity:
        print("✅ SAME PERSON & REAL HUMAN")
    elif not final_liveness and final_identity:
        print("⚠️  SAME PERSON but LIVENESS CHECK FAILED")
        print("   (May be a photo/video or poor lighting)")
    elif final_liveness and not final_identity:
        print("⚠️  REAL HUMAN but NOT the same person")
    else:
        print("❌ FAILED: Not the same person OR spoof detected")
        
    # Debugging info
    print("\n[DEBUG] Possible issues if failed:")
    if not final_identity:
        print("  • Different lighting between reference and live image")
        print("  • Different angles or facial expressions")
        print("  • Low camera quality or resolution")
    if not final_liveness:
        avg_texture = np.mean([r['metrics']['texture_score'] for r in live_results])
        avg_motion = np.mean([r['metrics']['motion_score'] for r in live_results])
        print(f"Texture score: {avg_texture:.1f} (need >{liveness.texture_threshold})")
        print(f"Motion score: {avg_motion:.1f} (need 2-80)")

    print("="*65)


if __name__ == "__main__":
    main()