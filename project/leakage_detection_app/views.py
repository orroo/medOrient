from django.shortcuts import render, redirect
from django.conf import settings
import os
from .utils import load_model, audio_to_spectrogram, predict
from PIL import Image
import torch
from torchvision import transforms
import os
from django.shortcuts import render, redirect
from django.conf import settings
from .utils import load_model
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # <-- Add this before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = os.path.join(settings.BASE_DIR, 'leakage_detection_app', 'model', 'cnn_model.pth')
model = load_model(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def leak_form(request):
    return render(request, 'leak.html')

def leak_predict(request):
    if request.method != 'POST' or 'file' not in request.FILES:
        return redirect('leak_form')

    # Save uploaded .wav file temporarily
    wav = request.FILES['file']
    tmp_dir = os.path.join(settings.MEDIA_ROOT, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    wav_path = os.path.join(tmp_dir, wav.name)

    with open(wav_path, 'wb') as f:
        for chunk in wav.chunks():
            f.write(chunk)

    # Generate spectrogram image
    spectrogram_path = wav_path.rsplit('.wav', 1)[0] + '.png'
    try:
        y, sr = librosa.load(wav_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        db = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(2.56, 2.56), dpi=50)
        librosa.display.specshow(db, sr=sr)
        plt.axis('off')
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return render(request, 'result_leak.html', {
            'leak_detected': False,
            'confidence': 0.0,
            'error': 'Failed to process audio.'
        })

    # Load and transform spectrogram image
    image = Image.open(spectrogram_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    label_map = {1: "No Leak", 0: "Leak"}
    print(f"Prediction: {label_map[prediction]} ({confidence*100:.2f}% confidence)")

    # Optional cleanup
    try:
        os.remove(spectrogram_path)
        os.remove(wav_path)
    except:
        pass

    return render(request, 'result_leak.html', {
        'leak_detected': bool(prediction == 0),
        
    })
