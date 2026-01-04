#test_voice.py
#  -----------------------------------
# REAL-TIME MIC → WAV2VEC2 INFERENCE
# -----------------------------------
import torch
import torch.nn as nn
import transformers
import librosa
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
import os 
# -----------------------------------
# CONFIG
# -----------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__),"best_wav2vec2_model_download.pt")
SAMPLE_RATE = 16000
RECORD_SECONDS = 3
TEMP_WAV = "temp_recording.wav"
MAX_LENGTH = 32007

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------------
# MODEL DEFINITION (SAME AS TRAINING)
# -----------------------------------
class CustomWav2Vec2Classifier(nn.Module):
    def __init__(self, hidden_dim=768, intermediate_dim=512, output_dim=2):
        super().__init__()
        self.wav2vec = transformers.Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )
        for p in self.wav2vec.parameters():
            p.requires_grad = False

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 4,
            num_heads=8,
            dropout=0.2,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(intermediate_dim // 2, output_dim),
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.wav2vec(x).last_hidden_state
        x = self.cnn_layers(x.transpose(1, 2))
        x, _ = self.self_attention(x.transpose(1, 2), x.transpose(1, 2), x.transpose(1, 2))
        x = x.mean(dim=1)
        return self.classifier(x)

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = CustomWav2Vec2Classifier().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✅ Model loaded")

# -----------------------------------
# FEATURE EXTRACTOR
# -----------------------------------
processor = transformers.Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base"
)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    features = processor(y, sampling_rate=sr, return_tensors="pt").input_values
    features = features.squeeze(0)

    if features.size(0) < MAX_LENGTH:
        features = torch.cat(
            [features, torch.zeros(MAX_LENGTH - features.size(0))], dim=0
        )
    else:
        features = features[:MAX_LENGTH]

    return features.unsqueeze(0).to(device)

# -----------------------------------
# RECORD FROM MICROPHONE
# -----------------------------------
def record_audio():
    print("\n🎤 Recording for 3 seconds... Speak now!")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    write(TEMP_WAV, SAMPLE_RATE, audio)
    print("✅ Recording complete")



def run_speech_inference():
    input("\n🎤 Press ENTER to record speech")
    record_audio()

    X = extract_features(TEMP_WAV)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)

    control_prob = probs[0, 0].item()
    dys_prob = probs[0, 1].item()

    THRESHOLD = 0.65

    if dys_prob > THRESHOLD:
        label = "Dysarthric"
        confidence = dys_prob
    elif control_prob > THRESHOLD:
        label = "Control"
        confidence = control_prob
    else:
        label = "Uncertain"
        confidence = max(control_prob, dys_prob)

    return {
    "speech_label": label,
    "speech_probabilities": {
        "Control": probs[0][0].item(),
        "Dysarthric": probs[0][1].item()
    }
}




# -----------------------------------
# MAIN LOOP
# -----------------------------------
if __name__ == "__main__":
    while True:
        input("\nPress ENTER to record (Ctrl+C to quit)")
        record_audio()

        X = extract_features(TEMP_WAV)

        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1)-0.1

        class_names = ["Control", "Dysarthric"]
        pred = torch.argmax(probs).item()

        print("Probabilities:", probs.cpu().numpy())
        print("🧠 Prediction:", class_names[pred])
