from functions import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image
import torch 



IMG_SIZE = (128, 128)  # adapte si nécessaire

def predict_image_class(img_path, encoder_model, classifier_model, label_encoder):
    img = load_img(img_path, target_size=IMG_SIZE)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Image testée")
    plt.show()

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    latent = encoder_model.predict(img_array)
    pred = classifier_model.predict(latent)

    class_index = np.argmax(pred)
    return label_encoder.inverse_transform([class_index])[0]



def kidney_label_to_medical_text(label):
    label = label.lower()

    if label == "normal":
        return (
            "L'analyse de l'image du rein ne montre pas d'anomalie évidente. "
            "L'aspect rénal semble globalement normal."
        )

    elif label in ["tumor", "cancer"]:
        return (
            "L'analyse de l'image du rein suggère la présence d'une masse rénale "
            "suspecte, pouvant être compatible avec une tumeur rénale."
        )

    elif label in ["cyst", "cystes"]:
        return (
            "L'analyse de l'image du rein suggère la présence de formations kystiques "
            "au niveau du rein."
        )

    elif label in ["stone", "calcul"]:
        return (
            "L'analyse de l'image du rein suggère la présence de calculs rénaux, "
            "compatibles avec une lithiase urinaire."
        )

    else:
        return (
            "L'analyse de l'image du rein n'a pas permis d'identifier clairement "
            "une anomalie spécifique."
        )


def run_kidney_image_query(
    img_path,
    encoder_model,
    classifier_model,
    label_encoder,
    faiss_idx,
    bm25_idx
):
    print("\n🖼️ Analyse de l'image du rein...")

    # 🔹 Étape 1 : prédiction image
    predicted_label = predict_image_class(
        img_path,
        encoder_model,
        classifier_model,
        label_encoder
    )

    print(f"[Modèle image] Classe prédite : {predicted_label}")

    # 🔹 Étape 2 : conversion en texte médical
    image_text = kidney_label_to_medical_text(predicted_label)

    print("\n[Vision → Texte médical]")
    print(image_text)

    # 🔹 Étape 3 : RAG
    passages = hybrid_search(image_text, faiss_idx, bm25_idx, k=TOP_K)
    context = "\n\n".join([p["text"] for p in passages])

    prompt = (
        SYSTEM_PROMPT + "\n\n"
        "Contexte médical récupéré :\n" + context[:4000] + "\n\n"
        "Résultat de l'analyse d'image du rein :\n" + image_text + "\n\n"
        "Explique la pathologie suspectée, les symptômes possibles, "
        "les examens complémentaires recommandés et la prise en charge générale. "
        "Précise la spécialité médicale à consulter."
    )

    out = llm_generate_api(
        prompt,
        model_name=HF_LLM_MODEL,
        max_tokens=500,
        temperature=0.0
    )

    print("\n[Assistant] Résultat basé sur l'image :\n")
    print(
        out.rstrip() +
        "\n\n⚠️ Cette information est fournie à titre informatif uniquement et ne constitue pas un diagnostic médical."
    )
    return out



def load_kidney_models(
    encoder_path,
    classifier_path,
    label_encoder_path
):
    """
    Charge les modèles du pipeline image rein (Keras + LabelEncoder)
    """
    from tensorflow.keras.models import load_model

    # 🔹 Modèles Keras
    encoder_model = load_model(encoder_path)
    classifier_model = load_model(classifier_path)

    # 🔹 Label encoder scikit-learn
    label_encoder = joblib.load(label_encoder_path)

    return encoder_model, classifier_model, label_encoder


encoder, classifier, le = load_kidney_models(
    encoder_path="./files/encoder.h5",
    classifier_path="./files/classifier.h5",
    label_encoder_path="./files/label_encoder.pkl"
)

peau_dir="./PEAU/"
# Charger les transformations
with open(peau_dir+"transform.pkl", "rb") as f:
    loaded_transform = pickle.load(f)

# Charger les classes
with open(peau_dir+"classes.pkl", "rb") as f:
    loaded_classes = pickle.load(f)

# with open(peau_dir+"efficientnet_weights.pth", "rb") as f:
#     loaded_model = pickle.load(f)

import torchvision.models as models

num_classes = len(loaded_classes)
loaded_model = models.efficientnet_b0(weights=None)  # ou efficientnet_b1, b2, etc.

# Modifier la dernière couche pour correspondre au nombre de classes
loaded_model.classifier[1] = torch.nn.Linear(
    loaded_model.classifier[1].in_features, 
    num_classes
)

# Charger les poids sauvegardés
loaded_model.load_state_dict(
    torch.load(peau_dir + "efficientnet_weights.pth", weights_only=True)
)

# Mettre le modèle en mode évaluation
loaded_model.eval()
print("✓ Modèle chargé et prêt")

# Charger le modèle



def predict_and_show_image(image_path, model, transform, class_names, device):
    """
    Affiche l'image et retourne la classe prédite avec confiance
    """
    model.eval()

    # Charger l'image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Prédiction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    predicted_class = class_names[pred_idx]
    confidence = probs[0][pred_idx].item()

    # # Affichage
    # plt.imshow(image)
    # plt.axis('off')
    # plt.title(f"Predicted: {predicted_class}")
    # plt.show()

    image_text = skin_label_to_medical_text(predicted_class)

    
    prompt = (
        SYSTEM_PROMPT + "\n\n"
        "Résultat de l'analyse d'image du rein :\n" + image_text + "\n\n"
        "Explique la pathologie suspectée, les symptômes possibles, "
        "les examens complémentaires recommandés et la prise en charge générale. "
        "Précise la spécialité médicale à consulter."
    )

    out = llm_generate_api(
        prompt,
        model_name=HF_LLM_MODEL,
        max_tokens=500,
        temperature=0.0
    )

    print("\n[Assistant] Résultat basé sur l'image :\n")
    print(
        out.rstrip() +
        "\n\n⚠️ Cette information est fournie à titre informatif uniquement et ne constitue pas un diagnostic médical."
    )
    return out




def skin_label_to_medical_text(label):
    label = label.lower()

    if label == "bkl":
        return (
            "L'analyse de l'image cutanée suggère la présence de lésions bénignes "
            "de type kératose, pouvant ressembler à des verrues ou à des taches pigmentées."
        )

    elif label == "nv":
        return (
            "L'analyse de l'image cutanée indique un nævus mélanocytaire, "
            "correspondant à un grain de beauté généralement bénin."
        )

    elif label == "df":
        return (
            "L'analyse de l'image cutanée suggère un dermatofibrome, "
            "une lésion cutanée bénigne, ferme, fréquemment localisée sur les membres inférieurs."
        )

    elif label == "mel":
        return (
            "L'analyse de l'image cutanée met en évidence une lésion suspecte "
            "compatible avec un mélanome, une forme agressive de cancer de la peau "
            "nécessitant une prise en charge médicale rapide."
        )

    elif label == "vasc":
        return (
            "L'analyse de l'image cutanée suggère une lésion vasculaire, "
            "liée à une anomalie des vaisseaux sanguins de la peau."
        )

    elif label == "bcc":
        return (
            "L'analyse de l'image cutanée suggère un carcinome basocellulaire, "
            "un cancer cutané fréquent à évolution généralement lente."
        )

    elif label == "akiec":
        return (
            "L'analyse de l'image cutanée suggère une kératose actinique "
            "ou un carcinome intraépithélial, une lésion précancéreuse liée à l'exposition solaire."
        )

    elif label == "ad":
        return (
            "L'analyse de l'image cutanée évoque une dermatite atopique, "
            "une affection inflammatoire chronique de la peau."
        )

    elif label == "fung":
        return (
            "L'analyse de l'image cutanée suggère une infection fongique, "
            "potentiellement liée à des champignons tels que les dermatophytes ou les levures."
        )

    elif label == "lp":
        return (
            "L'analyse de l'image cutanée évoque un lichen plan, "
            "une maladie inflammatoire pouvant affecter la peau et les muqueuses."
        )

    elif label == "vir":
        return (
            "L'analyse de l'image cutanée suggère une infection virale, "
            "comme des verrues ou un molluscum contagiosum."
        )

    else:
        return (
            "L'analyse de l'image cutanée n'a pas permis d'identifier clairement "
            "une pathologie spécifique."
        )
