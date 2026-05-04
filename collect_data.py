import cv2
import os

def collect_samples(label, num_samples=100):
    """Collect eye images for training"""
    # Try multiple indices for the camera
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Using camera index {index}")
            break
    else:
        print("Error: Could not open any camera.")
        return
    
    count = 0
    
    os.makedirs(f'data/{label}', exist_ok=True)
    
    print(f"Collecting {num_samples} samples for '{label}'")
    print("Press SPACE to capture, Q to quit")
    
    while count < num_samples:
        ret, frame = cap.read()
        cv2.putText(frame, f"{label}: {count}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Collect Data', frame)
        
        key = cv2.waitKey(1)
        if key == ord(' '):
            cv2.imwrite(f'data/{label}/{label}_{count}.jpg', frame)
            count += 1
            print(f"Captured {count}/{num_samples}")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Collect data
if __name__ == "__main__":
    collect_samples('eyes_open', 200)
    collect_samples('eyes_closed', 200)
    collect_samples('yawning', 100)
