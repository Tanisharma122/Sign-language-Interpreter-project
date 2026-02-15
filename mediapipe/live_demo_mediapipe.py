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

print(f"🎯 RTX 2050: {torch.cuda.is_available()}")

# Load LSTM model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = pickle.load(open('labels.pkl', 'rb'))

class SignLSTM(nn.Module):
    def __init__(self, num_classes=len(classes)):
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

model_state = torch.load('sign_language_lstm.pth', map_location=device, weights_only=False)
model = SignLSTM().to(device)
model.load_state_dict(model_state)
model.eval()

print(f"✅ RTX Model loaded! {len(classes)} classes: {classes}")

# SIGN TO HINDI MAPPING
word_mapping = {
    0: {"show": "HELLO - Namaste", "speak": "नमस्ते"},
    1: {"show": "NAME - Tanisha", "speak": "मैं तनिशा हूँ"}, 
    2: {"show": "NO - Nahi", "speak": "नहीं"},
    3: {"show": "PLEASE - Kripaya", "speak": "कृपया"},
    4: {"show": "SORRY - Maaf", "speak": "माफ़ करना"},
    5: {"show": "THANKS - Dhanyawad", "speak": "धन्यवाद"},
    6: {"show": "YES - Haan", "speak": "हाँ"}
}

def speak_hindi(hindi_text):
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=hindi_text, lang='hi', slow=False)
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# UI State
buffer = deque(maxlen=90)
sentence_words = []
current_detection = None
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🎤 RTX Sign → Hindi Speech - A=Add, S=Speak, C=Clear, Q=Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # MODERN DARK UI
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 40), -1)
    cv2.rectangle(overlay, (0, h-100), (w, h), (15, 15, 35), -1)
    
    # Hand Detection + LSTM Prediction
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    best_conf = 0
    best_cls = None
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        buffer.append(landmarks.flatten())
        
        if len(buffer) == 90:
            seq = np.array(list(buffer))
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(seq_tensor)
                conf, idx = torch.max(pred[0], 0)
                if conf > 0.5:
                    best_conf = conf.item()
                    best_cls = int(idx)
            buffer.popleft()
    
    # CURRENT DETECTION DISPLAY
    if best_cls is not None and best_cls in word_mapping:
        current_detection = word_mapping[best_cls]
        display_text = current_detection["show"]
        
        cv2.rectangle(frame, (20, 30), (500, 90), (0, 180, 255), 3)
        cv2.rectangle(frame, (25, 35), (495, 85), (255, 255, 255), -1)
        cv2.putText(frame, display_text, (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    else:
        cv2.putText(frame, "SHOW YOUR SIGN", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
    
    # SENTENCE DISPLAY
    if sentence_words:
        recent = sentence_words[-4:]
        sentence_display = " → ".join([word_mapping[int(w)]["show"].split(" - ")[0] for w in recent])
        
        cv2.rectangle(frame, (20, 130), (w-20, 220), (10, 10, 25), -1)
        cv2.putText(frame, "SENTENCE:", (35, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        text_lines = [sentence_display[i:i+35] for i in range(0, len(sentence_display), 35)]
        for i, line in enumerate(text_lines):
            cv2.putText(frame, line, (35, 185 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 200), 2)
    else:
        cv2.putText(frame, "Build sentence with 'A'", (35, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    
    # PROFESSIONAL CONTROLS
    controls = [
        ("A", "Add Word", (0, 255, 255)),
        ("S", "Speak Hindi", (255, 200, 0)), 
        ("C", "Clear", (100, 200, 255)),
        ("Q", "Quit", (255, 100, 100))
    ]
    
    for i, (key, text, color) in enumerate(controls):
        x = 50 + i * 220
        cv2.rectangle(frame, (x, h-85), (x+140, h-45), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, h-85), (x+140, h-45), color, 3)
        cv2.putText(frame, f"{key}: {text}", (x+10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Status indicator
    status_color = (0, 255, 0) if current_detection else (0, 100, 0)
    cv2.circle(frame, (w-60, 60), 20, status_color, -1)
    cv2.putText(frame, str(len(sentence_words)), (w-55, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    frame = cv2.addWeighted(frame, 0.9, overlay, 0.1, 0)
    cv2.imshow("🎤 RTX Sign Language → Hindi Speech", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    
    elif key == ord('a') and current_detection:
        sentence_words.append(str(best_cls))
        print(f"✅ Added: {current_detection['show']}")
    
    elif key == ord('c'):
        sentence_words.clear()
        current_detection = None
        print("🗑️ Cleared sentence")
    
    elif key == ord('s') and sentence_words:
        hindi_sentence = " ".join([word_mapping[int(w)]["speak"] for w in sentence_words])
        print(f"🗣️ Speaking: {hindi_sentence}")
        speak_hindi(hindi_sentence)

cap.release()
cv2.destroyAllWindows()
hands.close()
print("🎉 RTX Hindi Sign Language COMPLETE!")
