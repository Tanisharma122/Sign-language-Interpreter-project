import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import torch
import torch.nn as nn
import numpy as np
import pickle
from collections import deque
from gtts import gTTS
from playsound import playsound
import uuid
import mediapipe as mp

# Define the model class (must match the saved model structure)
class SignLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(63, 128, 2, bidirectional=True, dropout=0.3, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(hn)

class SignLanguageDetector:
    def __init__(self, model_path='sign_language_lstm.pth', labels_path='labels.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Device: {self.device}")
        
        # Load labels
        try:
            self.classes = pickle.load(open(labels_path, 'rb'))
            print(f"✅ Classes loaded: {self.classes}")
        except Exception as e:
            print(f"❌ Error loading labels: {e}")
            self.classes = []

        # Load Model
        try:
            self.model = SignLSTM(len(self.classes)).to(self.device)
            model_state = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(model_state)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        
        # State
        self.buffer = deque(maxlen=90)
        self.sentence_words = []
        self.current_detection = None
        self.cap = None

        self.word_mapping = {
            0: {"show": "HELLO - Namaste", "speak": "नमस्ते"},
            1: {"show": "NAME - Tanisha", "speak": "मैं तनिशा हूँ"}, 
            2: {"show": "NO - Nahi", "speak": "नहीं"},
            3: {"show": "PLEASE - Kripaya", "speak": "कृपया"},
            4: {"show": "SORRY - Maaf", "speak": "माफ़ करना"},
            5: {"show": "THANKS - Dhanyawad", "speak": "धन्यवाद"},
            6: {"show": "YES - Haan", "speak": "हाँ"}
        }

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def speak_hindi(self, hindi_text):
        try:
            filename = f"tts_{uuid.uuid4().hex}.mp3"
            tts = gTTS(text=hindi_text, lang='hi', slow=False)
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print(f"TTS Error: {e}")

    def generate_frames(self):
        if not self.cap:
            self.start_camera()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # UI Overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 40), -1)
            cv2.rectangle(overlay, (0, h-100), (w, h), (15, 15, 35), -1)
            
            # Hand Detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            best_cls = None
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                self.buffer.append(landmarks.flatten())
                
                if len(self.buffer) == 90 and self.model:
                    seq = np.array(list(self.buffer))
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        pred = self.model(seq_tensor)
                        conf, idx = torch.max(pred[0], 0)
                        if conf > 0.5:
                            best_cls = int(idx)
                    self.buffer.popleft()
            
            # Display Detection
            if best_cls is not None and best_cls in self.word_mapping:
                self.current_detection = self.word_mapping[best_cls]
                display_text = self.current_detection["show"]
                
                cv2.rectangle(frame, (20, 30), (500, 90), (0, 180, 255), 3)
                cv2.rectangle(frame, (25, 35), (495, 85), (255, 255, 255), -1)
                cv2.putText(frame, display_text, (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            else:
                cv2.putText(frame, "SHOW YOUR SIGN", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
            
            # Display Sentence
            if self.sentence_words:
                recent = self.sentence_words[-4:]
                sentence_display = " -> ".join([self.word_mapping[int(w)]["show"].split(" - ")[0] for w in recent])
                
                cv2.rectangle(frame, (20, 130), (w-20, 220), (10, 10, 25), -1)
                cv2.putText(frame, "SENTENCE:", (35, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                text_lines = [sentence_display[i:i+35] for i in range(0, len(sentence_display), 35)]
                for i, line in enumerate(text_lines):
                    cv2.putText(frame, line, (35, 185 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 200), 2)
            else:
                cv2.putText(frame, "Build sentence with 'A'", (35, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            
             # PROFESSIONAL CONTROLS (Visual only for stream)
            controls = [
                ("A", "Add Word", (0, 255, 255)),
                ("S", "Speak", (255, 200, 0)), 
                ("C", "Clear", (100, 200, 255)),
            ]
            
            for i, (key, text, color) in enumerate(controls):
                x = 50 + i * 220
                cv2.rectangle(frame, (x, h-85), (x+140, h-45), (0, 0, 0), -1)
                cv2.rectangle(frame, (x, h-85), (x+140, h-45), color, 3)
                cv2.putText(frame, f"{key}: {text}", (x+10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            frame = cv2.addWeighted(frame, 0.9, overlay, 0.1, 0)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        self.stop_camera()

if __name__ == "__main__":
    detector = SignLanguageDetector()
    detector.start_camera()
    # Simple test loop if run directly
    while True:
        try:
            next(detector.generate_frames())
        except KeyboardInterrupt:
            break
