# import os, torch, numpy as np
from functions import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import os 

import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),     # 64
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),    # 32
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),   # 16
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),  # 8
        )
        self.fc_mu = nn.Linear(256*8*8, latent_dim)
        self.fc_logvar = nn.Linear(256*8*8, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 256*8*8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 8, 8)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# class Classifier(nn.Module):
#     def __init__(self, latent_dim=64, num_classes=10):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(latent_dim, 128)
#         self.fc2 = nn.Linear(128, num_classes)
        
#     def forward(self, z):
#         x = torch.relu(self.fc1(z))
#         return self.fc2(x)
    
# class Classifier(nn.Module):
#     def __init__(self, latent_dim=64, num_classes=10):
#         super(Classifier, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(latent_dim, 128),           # net.0
#             nn.BatchNorm1d(128),                  # net.1
#             nn.ReLU(),                            # net.2
#             nn.Dropout(0.3),                      # net.3 (probably)
#             nn.Linear(128, 128),                  # net.4
#             nn.BatchNorm1d(128),                  # net.5
#             nn.ReLU(),                            # net.6
#             nn.Dropout(0.3),                      # net.7 (probably)
#             nn.Linear(128, num_classes)           # net.8
#         )
        
#     def forward(self, z):
#         return self.net(z)
    

class Classifier(nn.Module):
    def __init__(self, latent_dim=64, num_classes=1):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),      # net.0: 64 -> 256
            nn.BatchNorm1d(256),             # net.1
            nn.ReLU(),                       # net.2
            nn.Dropout(0.3),                 # net.3
            nn.Linear(256, 128),             # net.4: 256 -> 128
            nn.BatchNorm1d(128),             # net.5
            nn.ReLU(),                       # net.6
            nn.Dropout(0.3),                 # net.7
            nn.Linear(128, num_classes)      # net.8: 128 -> 1
        )
        
    def forward(self, z):
        return self.net(z)

vae = ConvVAE(latent_dim=64).to(device)
clf = Classifier(latent_dim=64, num_classes=1)





BUNDLE_PATH = "./VAE_CLF_bundle_acc_90_48 (1).pth"
assert os.path.exists(BUNDLE_PATH), "❌ Bundle 90.48 introuvable"
















bundle = torch.load(BUNDLE_PATH, map_location="cpu")
vae.load_state_dict(bundle["vae_state"], strict=True)
clf.load_state_dict(bundle["clf_state"], strict=True)

vae = vae.to(device).eval()
clf = clf.to(device).eval()

print("✅ Bundle 90.48 chargé")
print("🎯 Accuracy enregistrée dans le bundle:", bundle.get("accuracy", "N/A"))


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])



def predict_single_image(image_path, vae, clf, device):
    """
    Predict class for a single image
    
    Args:
        image_path: path to image file
        vae: loaded VAE model
        clf: loaded classifier model
        device: torch device (cpu or cuda)
    
    Returns:
        prediction: 0 or 1
        probability: confidence score
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Set models to eval mode
    vae.eval()
    clf.eval()
    
    with torch.no_grad():
        # Get latent representation
        mu, logvar = vae.encode(image_tensor)
        
        # Get prediction
        logit = clf(mu)
        prob = torch.sigmoid(logit).item()
        prediction = 1 if prob > 0.5 else 0
        result=f"Class: {'carrie' if prediction == 1 else 'normal'}"
    
    return result

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# prediction, probability = predict_single_image(
#     r"./envoyer\caries\4d0705fb-FA0235.jpg", 
#     vae, 
#     clf, 
#     device
# )

# print(f"Prediction: {prediction}")
# print(f"Probability: {probability:.4f}")
# print(f"Class: {'Positive' if prediction == 1 else 'Negative'}")