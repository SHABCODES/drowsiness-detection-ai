import cv2
import dlib
import numpy as np

class FaceDetector:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Define facial landmark indices
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
    
    def get_landmarks(self, frame):
        """Extract facial landmarks from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return None
        
        # Take first face
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Convert to numpy array
        coords = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        return coords
    
    def get_eye_regions(self, landmarks):
        """Extract left and right eye coordinates"""
        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        return left_eye, right_eye
    
    def get_mouth_region(self, landmarks):
        """Extract mouth coordinates"""
        return landmarks[self.MOUTH]
