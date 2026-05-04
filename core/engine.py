import cv2
import threading
import queue
import time
import yaml
import logging
import pygame
from utils.face_detector import FaceDetector
from utils.eye_utils import eye_aspect_ratio, mouth_aspect_ratio
import os

# quick logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="logs/app.log"
)

class DetectionEngine:
    def __init__(self, conf_file="config.yaml"):
        with open(conf_file, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.detector = FaceDetector()
        self.q = queue.Queue(maxsize=5)
        self.is_running = False
        
        # init audio
        try:
            pygame.mixer.init()
            # fallback if file is missing
            sound_file = self.cfg['alerts']['alert_sound_path']
            if os.path.exists(sound_file):
                pygame.mixer.music.load(sound_file)
        except Exception as e:
            print(f"Warn: Audio mixer couldn't start: {e}")

        # state counters
        self.alert_cnt = 0
        
    def _capture_loop(self):
        v_index = self.cfg['video']['camera_index']
        cam = cv2.VideoCapture(v_index)
        
        # apply res from config
        w, h = self.cfg['video']['resolution']
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        while self.is_running:
            ok, img = cam.read()
            if not ok:
                break
            
            if self.q.full():
                self.q.get() # drop old frame
            self.q.put(img)
            
        cam.release()

    def _process(self, img):
        pts = self.detector.get_landmarks(img)
        msg = "SYSTEM ACTIVE"
        lvl = 0 
        
        if pts is not None:
            l_eye, r_eye = self.detector.get_eye_regions(pts)
            mouth = self.detector.get_mouth_region(pts)
            
            ear = (eye_aspect_ratio(l_eye) + eye_aspect_ratio(r_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            
            # drowsiness check
            if ear < self.cfg['detection']['ear_threshold']:
                self.alert_cnt += 1
                if self.alert_cnt >= self.cfg['detection']['consecutive_frames']:
                    msg = "!!! DROWSY ALERT !!!"
                    lvl = 2
                    if self.cfg['alerts']['enable_sound'] and not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
            else:
                self.alert_cnt = 0

            # yawn check
            if mar > self.cfg['cfg']['mar_threshold'] if 'cfg' in self.cfg else 0.6:
                msg = "YAWN DETECTED"
                lvl = 1 if lvl < 2 else 2
                
        # UI overlays
        self._draw_hud(img, msg, lvl)
        return img

    def _draw_hud(self, img, msg, lvl):
        h, w = img.shape[:2]
        # header
        cv2.rectangle(img, (0, 0), (w, 50), (30, 30, 30), -1)
        # bottom bar
        colors = [(20, 180, 20), (0, 140, 255), (20, 20, 220)]
        cv2.rectangle(img, (0, h-40), (w, h), colors[lvl], -1)
        
        cv2.putText(img, msg, (15, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "DRIVER SAFETY AI", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def run(self):
        self.is_running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        
        print("Engine online. Press 'q' to exit.")
        while self.is_running:
            if not self.q.empty():
                frame = self.q.get()
                out = self._process(frame)
                cv2.imshow("System Feed", out)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = DetectionEngine()
    app.run()
