from django.shortcuts import render
from django.http import HttpResponse
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
import cv2
from PIL import Image
import numpy as np
import joblib  # 

# Create your views here.
def face(request):
    return render(request,"face.html")

model = ViTForImageClassification.from_pretrained("vit_pain_model")
extractor = ViTFeatureExtractor.from_pretrained("vit_pain_model")
le = joblib.load("vit_pain_model/label_encoder.pkl")  # optionnel
def start_camera(request):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            inputs = extractor(images=face_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                pred_id = logits.argmax(-1).item()
                pred_label = le.inverse_transform([pred_id])[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, pred_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36,255,12), 2)

        cv2.imshow('Webcam - Prédiction douleur', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return HttpResponse("Webcam fermée.")

