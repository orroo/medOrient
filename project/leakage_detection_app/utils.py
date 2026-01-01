# leakage_detection_app/utils.py

import os
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from .model_definition import LeakCNN  # Ensure this matches your model filename

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation (used during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Match training preprocessing
])

# ─── Load model ────────────────────────────────────────────────────────────────
def load_model(model_path):
    model = LeakCNN()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# ─── Convert .wav to Mel spectrogram image ─────────────────────────────────────
def audio_to_spectrogram(file_path, save_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        db = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(2.56, 2.56), dpi=50)  # 128x128 pixels
        librosa.display.specshow(db, sr=sr)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        raise

# ─── Predict from image ────────────────────────────────────────────────────────
def predict(model, image_path):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)
        probs = F.softmax(output, dim=1)[0].cpu().numpy()

    label = int(probs.argmax())     # 0 or 1
    confidence = float(probs[label])
    return label, confidence
