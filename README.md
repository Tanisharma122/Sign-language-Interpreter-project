# SignLink Pro - Real-time Sign Language Translator & 3D Avatar

![SignLink Pro Avatar](avatar_preview.png)

**SignLink Pro** is a cutting-edge real-time communication tool designed to bridge the gap between spoken language and sign language. Utilizing advanced computer vision (MediaPipe) and 3D rendering technologies, this project enables seamless bi-directional communication.

## 🌟 Key Features

### 1. Sign-to-Voice (Translation)
- **Real-time Detection**: Uses **MediaPipe** to detect hand gestures and body pose with high precision.
- **Instant Translation**: Converts recognized sign language gestures into text and spoken voice output.
- **Seamless UI**: A modern interface that displays the camera feed and translation results in real-time.

### 2. Voice-to-Sign (3D Avatar)
- **Interactive 3D Avatar**: Features a high-fidelity 3D character ("Avatar Girl") that performs sign language gestures.
- **Voice Interpretation**: Listens to spoken user input and instantly animates the avatar to sign the corresponding message.
- **GLB Model Integration**: smoothly renders complex 3D animations for realistic signing.

### 3. Premium User Interface
- **Modern Design**: Built with a "glassmorphism" aesthetic, featuring dark mode, sleek gradients, and responsive layouts.
- **Dashboard**: A central hub to access different modes (Learning, Translation, Profile).

## 🛠️ Technology Stack

- **Frontend**: 
  - HTML5, CSS3 (Custom Variables & Animations)
  - JavaScript (ES6+)
  - **Three.js** (for 3D Avatar rendering)
- **Backend**: 
  - Python 3.x
  - **Flask** (Web Server & API)
- **AI & Computer Vision**: 
  - **Google MediaPipe** (Hand & Pose Tracking)
  - OpenCV
- **Assets**: 
  - `.glb` 3D Models (Avatar animations)

## 📁 Project Structure

```bash
Sign_language_interpreter_project/
├── 3d-avatar-home-final/       # 3D Avatar Logic & Assets
├── integrated_avatar_app/      # Integrated Flask App
├── mediapipe/                  # AI/ML Detection Logic
├── Real-time-Sign-Language-Detection/ # Core Detection Scripts
├── index.html                  # Main Landing Page / UI
├── app.py                      # Flask Application Entry Point
└── README.md                   # Project Documentation
```

## 🚀 Getting Started

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Tanisharma122/Sign-language-Interpreter-project.git
    cd Sign-language-Interpreter-project
    ```

2.  **Install Dependencies**
    Ensure you have Python installed. Install the required packages:
    ```bash
    pip install flask opencv-python mediapipe
    ```

3.  **Run the Application**
    ```bash
    python app.py
    ```

4.  **Access in Browser**
    Open your browser and navigate to `http://localhost:5000` to start using SignLink Pro.

---

*Verified & maintained by Tanisha Sharma.*
