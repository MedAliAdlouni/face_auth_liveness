import cv2
from PIL import Image


def upload_image(*args):
    """
    Upload image either from webcam or from provided path.
    Returns a PIL.Image in RGB.
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
