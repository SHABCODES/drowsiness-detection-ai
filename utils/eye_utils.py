import numpy as np
from scipy.spatial import distance

def eye_aspect_ratio(eye):
    """
    Calculate Eye Aspect Ratio (EAR)
    Paper: "Real-Time Eye Blink Detection using Facial Landmarks"
    """
    # Compute euclidean distances between vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    
    # Compute euclidean distance between horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """
    Calculate Mouth Aspect Ratio (MAR)
    """
    # Vertical distances
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])   # 53, 57
    
    # Horizontal distance
    C = distance.euclidean(mouth[0], mouth[6])   # 49, 55
    
    mar = (A + B) / (2.0 * C)
    return mar
