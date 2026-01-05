



# def load_model(path):
#     model = PneumoniaVAE()
#     model.load_state_dict(torch.load(path))
#     model.eval()
#     return model

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms



MODEL_DIR ="./model_pneumonia_vae.pth"




device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device :", device)

# Recreate the exact same architecture
num_classes = 5  # Make sure this matches your training setup
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers (same as training)
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer with dropout (same as training)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, num_classes)
)

# Load the saved checkpoint
checkpoint = torch.load(MODEL_DIR, map_location=device)

# Check what's in the checkpoint
print("Keys in checkpoint:", checkpoint.keys())

# Load the classifier state (the ResNet model)
if 'clf_state' in checkpoint:
    model.load_state_dict(checkpoint['clf_state'])
    print("Modèle chargé depuis 'clf_state'!")
    
    # Print additional info if available
    if 'accuracy' in checkpoint:
        print(f"Model accuracy: {checkpoint['accuracy']}")
    if 'date' in checkpoint:
        print(f"Saved on: {checkpoint['date']}")
else:
    # If the structure is different, try loading directly
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()  # Set to evaluation mode
print("Modèle prêt pour l'inférence!")




# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                        std=[0.229, 0.224, 0.225])
# ])


img_size = 224  # tu peux monter après à 320/384 si possible

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


def predict_image(image_path, model, device):
    """Predict class for a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()


class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']


def predict_and_display(image_path):
    """Predict and display results for an image"""
    predicted_class, confidence, probs = predict_image(image_path, model, device)
    
    print(f"\n{'='*50}")
    print(f"Image: {image_path}")
    print(f"{'='*50}")
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nAll Class Probabilities:")
    for i, (name, prob) in enumerate(zip(class_names, probs)):
        print(f"  {name:25s}: {prob:.2%}")
    print(f"{'='*50}\n")
    
    return class_names[predicted_class]


# img_path =r".\envoyer\lung desease\normal\06.jpeg"

# predicted_class, confidence, probs = predict_and_display(img_path)

# print(predicted_class, confidence, probs)