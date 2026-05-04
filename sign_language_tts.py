import cv2
import mediapipe as mp
import pyttsx3
import time

class SignLanguageInterpreter:
    def __init__(self):
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # TTS initialization
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Gestures logic
        self.last_sign = ""
        self.current_word = ""
        self.cooldown = 0
        
    def speak(self, text):
        """Speak text asynchronously"""
        self.engine.say(text)
        self.engine.runAndWait()

    def get_gesture(self, landmarks):
        """
        Simple heuristic-based gesture detection (Alphabet/Signs)
        In a production environment, you'd use a CNN here, 
        but we can use landmark positions for common signs.
        """
        # Get finger tips and bases
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Simple Logic for a few signs:
        # 1. 'Hello' (Open Palm)
        if index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y:
            return "HELLO"
        
        # 2. 'Peace' (V-Sign)
        if index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y and ring_tip.y > landmarks[14].y:
            return "PEACE"

        # 3. 'Like' (Thumbs up)
        if thumb_tip.y < landmarks[3].y and index_tip.x > landmarks[6].x:
            return "LIKE"
            
        return "UNKNOWN"

    def run(self):
        cap = cv2.VideoCapture(0)
        print("Starting Sign Language Interface...")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            gesture = "NO HANDED"
            
            # Draw HUD
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (45, 45, 45), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks (Professional thin lines)
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Detect Gesture
                    gesture = self.get_gesture(hand_landmarks.landmark)
                    
                    # Prevent rapid repeat speaking
                    if gesture != "UNKNOWN" and gesture != self.last_sign and time.time() - self.cooldown > 2:
                        self.last_sign = gesture
                        self.cooldown = time.time()
                        print(f"Detected: {gesture}")
                        # self.speak(gesture) # Uncomment to enable active speaking
            
            # Display HUD Text
            cv2.putText(frame, f"SIGN DETECTED: {gesture}", (20, 50),
                       cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Sign Language to Speech", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    interpreter = SignLanguageInterpreter()
    interpreter.run()
