from django.shortcuts import render
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from django.core.files.storage import default_storage
import joblib
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from uuid import uuid4
from django.conf import settings  # pour MEDIA_URL
from django.core.files.storage import default_storage
def Kidney(request):
    return render(request,"kidney.html")
# Charger les modèles
encoder = load_model('./saved-model/encoder.h5')
classifier = load_model('./saved-model/classifier.h5')
le = joblib.load('./saved-model/label_encoder.pkl')

IMG_SIZE = (128, 128)

def predict_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        
        # Sauvegarder l'image
        img_file = request.FILES['image']
        img_name = str(uuid4()) + "_" + img_file.name
        img_path = default_storage.save(f"tmp/{img_name}", img_file)
        full_path = os.path.join(settings.MEDIA_ROOT, img_path)

        # Prétraitement image
        img = load_img(full_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array_exp = np.expand_dims(img_array, axis=0)

        # Prédiction
        latent = encoder.predict(img_array_exp)
        pred = classifier.predict(latent)
        class_index = np.argmax(pred)
        class_name = le.inverse_transform([class_index])[0]

        # LIME
        def lime_predict_fn(images):
            images = np.array(images) / 255.0
            latent = encoder.predict(images)
            return classifier.predict(latent)

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image=(img_array * 255).astype('double'),
            classifier_fn=lime_predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        print("Image reçue:", img_file.name)
        print("Classe prédite:", class_name)

        # Enregistrer l’image LIME
        lime_name = f"lime_{img_name}.png"
        lime_rel_path = f"tmp/{lime_name}"
        lime_abs_path = os.path.join(settings.MEDIA_ROOT, lime_rel_path)
        plt.imsave(lime_abs_path, mark_boundaries(temp / 255.0, mask))

        # Renvoyer le HTML avec images visibles via MEDIA_URL
        return render(request, 'kidney.html', {
            'prediction': class_name,
            'image_url': settings.MEDIA_URL + img_path,
            'lime_url': settings.MEDIA_URL + lime_rel_path
        })
