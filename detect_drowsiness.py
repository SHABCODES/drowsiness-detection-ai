import cv2
import torch
import numpy as np
from utils.face_detector import FaceDetector
from utils.eye_utils import eye_aspect_ratio, mouth_aspect_ratio
from models.drowsiness_cnn import DrowsinessNet
import pygame
from collections import deque
import os

class DrowsinessDetector:
    def __init__(self, model_path='best_model.pth'):
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DrowsinessNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Face detector
        self.face_detector = FaceDetector()
        
        # Thresholds
        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.6
        self.CONSEC_FRAMES = 20
        
        # Counters
        self.frame_counter = 0
        self.ear_history = deque(maxlen=30)
        
        # Alert system
        pygame.mixer.init()
        self.alert_sound = self.create_beep()
        
        # Status
        self.is_drowsy = False
    
    def create_beep(self):
        """Create alert sound"""
        # Simple beep (you can replace with actual audio file)
        pygame.mixer.music.load('alert.wav') if os.path.exists('alert.wav') else None
        return True
    
    def preprocess_eye(self, frame, eye_coords):
        """Extract and preprocess eye region"""
        # Get bounding box
        x, y, w, h = cv2.boundingRect(eye_coords)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2*padding)
        h = min(frame.shape[0] - y, h + 2*padding)
        
        # Extract region
        eye_region = frame[y:y+h, x:x+w]
        
        # Resize and normalize
        eye_region = cv2.resize(eye_region, (64, 64))
        eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        eye_tensor = torch.from_numpy(eye_region).float().unsqueeze(0).unsqueeze(0)
        eye_tensor = (eye_tensor / 255.0 - 0.5) / 0.5
        
        return eye_tensor.to(self.device)
    
    def detect(self, frame):
        """Main detection logic"""
        # Get facial landmarks
        landmarks = self.face_detector.get_landmarks(frame)
        
        if landmarks is None:
            return frame, "No face detected"
        
        # Get eye and mouth regions
        left_eye, right_eye = self.face_detector.get_eye_regions(landmarks)
        mouth = self.face_detector.get_mouth_region(landmarks)
        
        # Calculate EAR and MAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)
        
        self.ear_history.append(ear)
        
        # COMMENTED OUT: Draw landmarks for a cleaner look
        # for (x, y) in left_eye:
        #     cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        # for (x, y) in right_eye:
        #     cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        # for (x, y) in mouth:
        #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        # Drowsiness detection
        status = "DRIVING MODE: ACTIVE"
        color = (0, 255, 0)
        
        # Check EAR threshold
        if ear < self.EAR_THRESHOLD:
            self.frame_counter += 1
            
            if self.frame_counter >= self.CONSEC_FRAMES:
                status = "DROWSY ALERT - PULL OVER!"
                color = (0, 0, 255)
                self.is_drowsy = True
                
                # Play alert
                if pygame.mixer.get_init():
                    try:
                        pygame.mixer.music.play()
                    except:
                        pass
        else:
            self.frame_counter = 0
            self.is_drowsy = False
        
        # UI Styling - Semi-transparent header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Check for yawning
        if mar > self.MAR_THRESHOLD:
            cv2.putText(frame, "FATIGUE WARNING: YAWNING", (frame.shape[1] - 350, 40),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 165, 255), 1)
        
        # Display metrics with professional fonts
        cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f}", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        
        # Banner for Status
        # Draw status background based on alert state
        banner_color = (0, 255, 0) if not self.is_drowsy else (0, 0, 255)
        cv2.rectangle(frame, (0, frame.shape[0]-50), (frame.shape[1], frame.shape[0]), banner_color, -1)
        
        cv2.putText(frame, status, (20, frame.shape[0] - 15),
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        return frame, status
    
    def run(self):
        """Run real-time detection"""
        cap = cv2.VideoCapture(0)
        
        print("Starting drowsiness detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect drowsiness
            frame, status = self.detect(frame)
            
            # Display
            cv2.imshow('Drowsiness Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DrowsinessDetector('best_model.pth')
    detector.run()
