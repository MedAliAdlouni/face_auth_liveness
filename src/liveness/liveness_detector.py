import cv2
import numpy as np
from collections import deque
import time

from src.config import TEXTURE_THRESHOLD

class LivenessDetector:
    def __init__(self):
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
                
        # More strict thresholds
        self.texture_threshold = TEXTURE_THRESHOLD  # Read from config instead of hardcoded
        self.motion_history = deque(maxlen=30)
        self.consecutive_closed = 0
        
        # Color analysis history
        self.color_variations = deque(maxlen=30)
        
    def calculate_texture_score(self, face_roi):
        """Calculate texture complexity - real skin has more variation"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Multiple texture measures
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Local binary pattern-like measure
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_score = edge_magnitude.std()
        
        # Combined score
        texture_score = laplacian_var + edge_score
        return texture_score
    
    def detect_motion(self, face_coords):
        """Detect natural micro-movements with stricter requirements"""
        if len(self.motion_history) < 10:
            self.motion_history.append(face_coords)
            return 0, False
        
        self.motion_history.append(face_coords)
        
        # Calculate movement variance
        positions = np.array(list(self.motion_history))
        variance = np.var(positions, axis=0).sum()
        
        # Check for natural movement pattern (not too still, not too erratic)
        is_natural = 2 < variance < 80
        
        return variance, is_natural
    
    def analyze_color_variation(self, face_roi):
        """Real faces have color variation due to blood flow"""
        # Convert to different color space
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
        
        # Calculate color channel variations
        h_var = np.var(hsv[:,:,0])
        cr_var = np.var(ycrcb[:,:,1])
        
        color_score = h_var + cr_var
        self.color_variations.append(color_score)
        
        # Real faces should have consistent color variation
        if len(self.color_variations) > 20:
            variation_consistency = np.std(list(self.color_variations))
            return color_score, variation_consistency < 50
        
        return color_score, False
 
    def check_screen_reflection(self, face_roi):
        """Detect screen reflections and uniform lighting typical of fake images"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Check for uniform brightness (screens often have this)
        brightness_std = np.std(gray)
        
        # Check for regular patterns (screen pixels)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        # Screens often have periodic patterns
        spectrum_peaks = np.sort(magnitude_spectrum.flatten())[-100:].mean()
        
        # Low brightness variation + high spectrum peaks = likely screen
        is_screen_like = brightness_std < 30 and spectrum_peaks > 10000
        
        return brightness_std, is_screen_like
    
    def is_real_face(self, frame, face_coords):
        """Determine if face is real with stricter criteria"""
        x, y, w, h = face_coords
        face_roi = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Test 1: Texture analysis (stricter)
        texture_score = self.calculate_texture_score(face_roi)
        texture_passed = texture_score > self.texture_threshold
        
        # Test 2: Motion analysis (stricter)
        motion_score, motion_natural = self.detect_motion((x, y))
                
        # Test 3: Color variation analysis
        color_score, color_natural = self.analyze_color_variation(face_roi)
        
        # Test 4: Screen reflection detection
        brightness_std, is_screen = self.check_screen_reflection(face_roi)
        screen_passed = not is_screen
        
        # Stricter scoring system
        score = 0
        confidence_factors = []
        
        if texture_passed:
            score += 2  # Texture is very important
            confidence_factors.append("texture")
        
        if motion_natural:
            score += 2  # Natural motion is crucial
            confidence_factors.append("motion")
        
        if color_natural:
            score += 1
            confidence_factors.append("color")
        
        if screen_passed:
            score += 1
            confidence_factors.append("no_screen")
        else:
            score -= 2  # Penalize screen-like characteristics
        
        # Need at least 5 points to be considered real (stricter)
        is_real = score >= 5
        
        return is_real, {
            'texture_score': texture_score,
            'texture_passed': texture_passed,
            'motion_score': motion_score,
            'motion_natural': motion_natural,
            'color_score': color_score,
            'brightness_std': brightness_std,
            'is_screen_like': is_screen,
            'total_score': score,
            'max_score': 9,
            'confidence_factors': confidence_factors
        }


