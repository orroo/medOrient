import os
import torch
import torch.nn as nn
import librosa
import numpy as np

# Configuration constants
SR = 44100
N_MELS = 64
WINDOW_LENGTH = 1024
HOP_LENGTH = 512
DURATION = 5.0  # seconds, fixed audio clip length for feature extraction
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_vae.pth')

# VAE Model Definition
class VAE(nn.Module):
    def __init__(self, input_dim=N_MELS, hidden_dim=32, latent_dim=16):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# Load the pretrained VAE model (CPU only here, adjust if needed)
device = torch.device("cpu")
vae_model = VAE()
vae_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
vae_model.to(device).eval()

def extract_features(file_path):
    """
    Load audio file (fixed duration) and extract mean Mel-spectrogram vector.
    Returns None if audio is too short.
    """
    y, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    # Check if audio length is sufficient
    if len(y) < SR * DURATION * 0.9:  # 90% threshold to allow slight variation
        return None

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=WINDOW_LENGTH,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feature_vector = mel_db.mean(axis=1)
    return feature_vector

def compute_day_error(file_path):
    """
    Extract features from audio, compute reconstruction MSE from VAE.
    Returns None if audio too short or feature extraction fails.
    """
    features = extract_features(file_path)
    if features is None:
        return None

    x = torch.from_numpy(features).float().to(device)
    with torch.no_grad():
        recon, _, _ = vae_model(x)
        mse = ((recon - x) ** 2).mean().item()
    return mse
