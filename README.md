# Driver Fatigue & Safety Monitor AI 🚗💨

A production-grade, real-time safety system that monitors driver behavior using Computer Vision and Deep Learning. This project detects drowsiness and fatigue levels to prevent accidents.

## 🌟 Key Features
- **Real-Time Drowsiness Detection**: Uses Eye Aspect Ratio (EAR) with a custom CNN for high accuracy.
- **Fatigue Analysis**: Tracks yawning patterns via Mouth Aspect Ratio (MAR).
- **Multi-Threaded Architecture**: Optimized performance for smooth 30+ FPS processing.
- **Modern HUD**: Professional "Heads-Up Display" user interface.
- **Sign Language Integration**: Gesture-to-speech module for enhanced accessibility.
- **ONNX Optimized**: 3x faster inference for edge deployment.

## 🛠️ Tech Stack
- **Languages**: Python
- **AI Frameworks**: PyTorch, MediaPipe
- **Computer Vision**: OpenCV, dlib
- **Optimization**: ONNX Runtime
- **Utilities**: Pygame (Audio), pyttsx3 (TTS)

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/drowsiness-detection.git
cd drowsiness-detection
```

### 2. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Initialize Assets
```bash
python download_models.py
python generate_beep.py
```

### 4. Run the Monitor
```bash
python main.py
```

## 📊 Performance
- **CNN Accuracy**: ~94% on test set
- **Latency**: 1.22ms (ONNX) / 3.95ms (PyTorch)
- **Speedup**: 3.24x via ONNX quantization

## 👤 Author
- **[Your Name]** - *Project Lead & AI Engineer*
