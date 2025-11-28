from PIL import Image
import os
import time
import pickle
import cv2
import traceback
traceback.print_exc()

from src.detectors.face_detector import detect_face
from src.embedding.face_embedding import extract_face_embedding
from src.utils.helpers import get_embedding_path, upload_image


def register_pipeline(first_name, last_name, img):
    """
    Registration pipeline: Register a new user by extracting and saving their face embedding.
    
    Pipeline steps:
        1. Detect face in the image
        2. Extract face embedding
        3. Save embedding to file (first_name_last_name.pkl)
    
    Args:
        first_name: User's first name
        last_name: User's last name
        image: PIL image
        
    Returns:
        dict: Registration results including success status and embedding info
    """
    try:
        print("USER REGISTRATION PIPELINE")
        print(f"\nRegistering: {first_name} {last_name}")
                
        # Step 1: Detect face
        print("\n[1/3] Detecting face...")
        face_tensor, detection_prob = detect_face(img)
        
        # if no face detected or low confidence -> raise error
        if face_tensor is None or detection_prob < 0.5:
            raise ValueError("No face detected in the image. Please provide a clear face image.")
        
        print(f"✓ Face detected (confidence: {detection_prob:.4f})")
        
        # Step 2: Extract embedding
        print("\n[2/3] Extracting face embedding...")
        embedding = extract_face_embedding(face_tensor)
        print(f"✓ Embedding extracted (shape: {embedding.shape})")
        
        # Step 3: Save embedding
        print("\n[3/3] Saving embedding...")
        embedding_path = get_embedding_path(first_name, last_name)
                
        # Save embedding to file
        user_data = {
            'first_name': first_name,
            'last_name': last_name,
            'embedding': embedding,
            'registration_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(embedding_path, 'wb') as f:
            pickle.dump(user_data, f)
        
        print(f"✓ Embedding saved to: {embedding_path}")
        
        result = {
            'success': True,
            'user': f"{first_name} {last_name}",
            'embedding_path': embedding_path,
            'detection_probability': detection_prob,
            'embedding_shape': embedding.shape
        }
        return result
        
    except Exception as e:
        print(f"\n REGISTRATION FAILED: {str(e)}")
        return {
            'success': False,
            'message': str(e),
            'user': f"{first_name} {last_name}"
        }

if __name__ == "__main__":
    first_name = input("Enter first name: ").strip()
    last_name = input("Enter last name: ").strip()
    
    image = upload_image()
    
    result = register_pipeline(first_name, last_name, image)
    
    if result['success']:
        print(f"\n{first_name} {last_name} registered successfully!")
    else:
        print(f"\n Registration failed: {result['message']}")
    