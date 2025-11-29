import cv2
import time
import numpy as np
import logging

from src.liveness.liveness_detection import LivenessDetector

logger = logging.getLogger(__name__)


def liveness_detection(duration=10):
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

        captured_frames.append(frame.copy())

        faces = detector.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        if len(faces) > 0:
            face_detected_frames += 1

        for (x, y, w, h) in faces:
            is_real, metrics = detector.is_real_face(frame, (x, y, w, h))
            all_results.append({'is_real': is_real, 'metrics': metrics})

            color = (0, 255, 0) if is_real else (0, 0, 255)
            label = "REAL" if is_real else "FAKE/SPOOF"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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
    liveness_passed = False
    liveness_confidence = 0.0

    if all_results:
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

    return liveness_passed, liveness_confidence


def perform_liveness(liveness_duration=10):
    logger.info("Please look at the camera and move naturally.\n")
    detector, all_results, captured_frames = liveness_detection(duration=liveness_duration)
    liveness_passed, liveness_confidence = analyze_liveness_results(all_results, detector)
    return liveness_passed, liveness_confidence, detector, all_results, captured_frames
