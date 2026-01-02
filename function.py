import requests

import faiss

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd



import httpx
import json
from openai import OpenAI
import re

from dotenv import load_dotenv

import os



load_dotenv()


EMBEDDING_MODEL_NAME=os.getenv("EMBEDDING_MODEL_NAME")

OCR_URL=os.getenv("OCR_URL")

ESPRIT_API_KEY=os.getenv("ESPRIT_API_KEY")

LLM_URL=os.getenv("LLM_URL")

# URL de ton API OCR

# Chemin de l'image à envoyer

# # Ouvrir l’image
# with open(image_path, "rb") as f:
#     files = {"image": (image_path, f, "image/jpeg")}
#     response = requests.post(url, files=files)

# # Affichage du résultat
# print("Status:", response.status_code)
# print("Réponse OCR:", response.json())



# with open(image_path, "rb") as f:
#     data = {"image": (image_path, f, "/content/3.jpg")}
#     resp = requests.post(url, files=data)

# print(resp.json()["text"])
import re
import json
import httpx
from openai import OpenAI


def extract_text(path):
    with open(path, "rb") as f:
        data = {"image": (path, f, "/content/3.jpg")}
        resp = requests.post(OCR_URL, files=data)

    print(resp.json()["text"])
    return resp.json()["text"]



# --- Connexion à l’API TokenFactory --- 
# Utilisation de httpx pour la connexion HTTP avec TokenFactory (Llama)

# --- Connexion à l’API ESPRIT ---
http_client = httpx.Client(verify=False)
client = OpenAI(
    api_key=ESPRIT_API_KEY,
    base_url=LLM_URL,
    http_client=http_client
)


def correct_medicine_name_with_llama(med_name, dosage, duration):
    """
    Appelle llama pour valider ou corriger un nom de médicament,
    en utilisant le dosage et la durée comme contexte pour plus de précision.
    """
    # Utilisation de f-strings pour inclure les détails contextuels
    prompt = f"""
    Vous êtes un expert en pharmacie.

    VOTRE TÂCHE UNIQUE :
    Corriger éventuellement le nom d’un médicament extrait d’une ordonnance OCR.

    RÈGLES STRICTES (OBLIGATOIRES) :
    - Répondez avec UN SEUL nom de médicament.
    - AUCUNE phrase.
    - AUCUNE explication.
    - AUCUNE ponctuation.
    - AUCUNE parenthèse.
    - AUCUNE alternative.
    - AUCUN commentaire.

    - Le nom fourni peut contenir des erreurs OCR.
    - La posologie et la durée servent UNIQUEMENT d’indice contextuel léger.
    - Si le nom est correct, retournez-le tel quel.
    - Si le nom est mal orthographié, proposez la correction la plus probable.
    - Si vous avez le moindre doute, retournez le nom original EXACTEMENT tel qu’il est fourni.
    - Ne remplacez JAMAIS un médicament par un autre différent.
    - Ne déduisez JAMAIS une molécule à partir du contexte.

    IMPORTANT :
    - If the input contains extra words or noise, focus ONLY on the drug name part.
    - If the cleaned name does not clearly match a known drug, return it unchanged.

    Informations de l’ordonnance :
    - Nom extrait (OCR brut) : {med_name}
    - Posologie / Instructions : {dosage}
    - Durée : {duration}

    Nom du médicament corrigé ou confirmé :

    """

    # Envoi de la requête via l'API ESPRIT
    try:
        response = client.chat.completions.create(
            model="hosted_vllm/Llama-3.1-70B-Instruct",
            messages=[ 
                {"role": "system", "content": "Assistant médical strict, factuel, sans spéculation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=700,
            top_p=0.9
        )

        # Log de la réponse pour inspecter sa structure
        print("Réponse complète de l'API:", response)

        # Accéder à la réponse en extrayant les choix correctement
        content = response.choices[0].message.content  # Utiliser la bonne syntaxe pour accéder au contenu
        return content

    except Exception as e:
        print(f"⚠️ Erreur lors de la correction du médicament '{med_name}' : {e}")
        return {"drug": med_name, "error": str(e)}




def extract_and_correct_meds_with_llama(ocr_text):
    """
    Appelle llama pour extraire les médicaments, la posologie et la durée
    d'un texte OCR. Concentré sur l'extraction brute et le format JSON strict.
    """
    prompt = prompt = f"""
    Vous êtes un système d’extraction d’informations à partir d’un texte OCR d’ordonnance médicale.

    VOTRE TÂCHE UNIQUE :
    Extraire les médicaments EXACTEMENT tels qu’ils apparaissent dans le texte OCR,
    ainsi que leur posologie et leur durée si présentes.

    RÈGLES STRICTES :
    - Ne corrigez PAS l’orthographe.
    - Ne devinez PAS le nom correct.
    - Ne choisissez PAS le médicament “le plus probable”.
    - Ne normalisez RIEN.
    - Ne validez RIEN médicalement.
    - Conservez les mots EXACTEMENT tels qu’ils apparaissent dans le texte OCR.

    - Le texte peut contenir des fautes OCR, des mots tronqués ou fusionnés.
    - Si une information est absente, utilisez "".

    FORMAT DE SORTIE OBLIGATOIRE :
    Vous devez répondre UNIQUEMENT avec un JSON valide.
    AUCUN texte avant ou après.
    AUCUNE explication.

    FORMAT EXACT :
    [
    {{
        "drug": "",
        "dosage": "",
        "duration": ""
    }}
    ]

    SI AUCUN MÉDICAMENT N’EST TROUVÉ :
    Retournez exactement []

    TEXTE OCR :
    {ocr_text}
    """
    # Envoi de la requête via l'API ESPRIT
    try:
        response = client.chat.completions.create(
            model="hosted_vllm/Llama-3.1-70B-Instruct",
            messages=[ 
                {"role": "system", "content": "Assistant médical strict, factuel, sans spéculation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=700,
            top_p=0.9
        )

        # Log de la réponse pour inspecter sa structure
        print("Réponse complète de l'API:", response)

        # Accéder à la réponse en extrayant le contenu JSON depuis la réponse de l'API
        response_text = response.choices[0].message.content  # Utiliser la bonne syntaxe pour accéder au contenu
        print("Réponse textuelle extraite :", response_text)

        # Vérifier si la réponse est bien un JSON valide
        try:
            print(f"🔍 Response text type: {type(response_text)}")
            print(f"🔍 Response text length: {len(response_text) if response_text else 0}")
            print(f"🔍 Response text content: '{response_text}'")
            print(f"🔍 First 100 chars: '{response_text[:100] if response_text else 'EMPTY'}'")
            
            extracted_meds = json.loads(response_text)
        except json.JSONDecodeError as json_err:
            print(f"⚠️ Erreur lors du décodage JSON : {json_err}")
            print(f"⚠️ Contenu reçu : {repr(response_text)}")
            return {"error": "Réponse API invalide", "details": str(response_text)}

        return extracted_meds

    except Exception as e:
        print(f"⚠️ Erreur lors de l'extraction et correction des médicaments : {e}")
        return {"error": str(e)}






# # --- EXEMPLE D'UTILISATION ---
# ocr_text_input = "Rx Tab Auguentin 625mg x5day Enzoflarn 5days PaniD 40mg before meals Hexigel gum paste 1week"

# # Extrait et corrige les médicaments extraits via Llama
# extracted_meds = extract_and_correct_meds_with_llama(ocr_text_input)

# # Vérifier si la réponse est bien un tableau de médicaments
# if isinstance(extracted_meds, list):
#     # Correction supplémentaire avec Llama si nécessaire
#     corrected_meds = []
#     for item in extracted_meds:
#         corrected_name = correct_medicine_name_with_llama(item['drug'], item['dosage'], item['duration'])
#         corrected_meds.append({
#             "drug": corrected_name,
#             "dosage": item['dosage'],
#             "duration": item['duration']
#         })

#     # Afficher les résultats
#     print("\n--- RÉSULTAT FINAL ---")
#     print(json.dumps(corrected_meds, indent=2, ensure_ascii=False))
# else:
#     print(f"⚠️ Erreur : {extracted_meds.get('error', 'Erreur inconnue')}")






# ALWAYS CPU
try : 
    embedding_model = SentenceTransformer('models/'+EMBEDDING_MODEL_NAME)
    print( "loaded locally")
except :
    # If not found, load the model from the internet and save it locally
    embedding_model = SentenceTransformer( "sentence-transformers/"+EMBEDDING_MODEL_NAME
    )
    print("saving")
    embedding_model.save('models/'+EMBEDDING_MODEL_NAME)  # Save locally
    print("saved")

def compute_embeddings(text_list):
    embeddings = embedding_model.encode(text_list, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")





def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cos similarity
    index.add(embeddings)
    return index



def match_drug_embeddings(drug_name, df, index, embeddings, threshold=0.80):

    query_emb = compute_embeddings([drug_name])  # (1, 384)

    scores, indices = index.search(query_emb, k=1)
    score = scores[0][0]
    idx = indices[0][0]

    if score >= threshold:
        return df["canonical"].iloc[idx], score

    return None, score





# # import pandas as pd
# df = pd.read_csv(r"C:\Users\hp\Desktop\4ème\projet\medicaments_clean_for_ocr.csv")

# # Compute dataset embeddings
# dataset_embeddings = compute_embeddings(df["canonical"].tolist())

# # Build index
# index = create_faiss_index(dataset_embeddings)

# # Query
# drug = "ENZOFLan"
# match, score = match_drug_embeddings(drug, df, index, dataset_embeddings)

# print(match, score)


# # --- Connexion API TokenFactory ---
# esprit_api_key = "sk-e376096028c847389e18f6d1f650be93"

# http_client = httpx.Client(verify=False)
# client = OpenAI(
#     api_key=esprit_api_key,
#     base_url="https://tokenfactory.esprit.tn/api",
#     http_client=http_client
# )


# ------------------------------------------------------
# 🛡️ 1. JSON Cleaner : retire le markdown, répare virgules
# ------------------------------------------------------
def clean_json_output(text):
    text = text.strip()

    # Enlever éventuels ```json ... ```
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```", "", text).strip()

    # Retirer trailing commas avant les }
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    return text


# ------------------------------------------------------
# 🛡️ 2. VALIDATION DES CHAMPS
# ------------------------------------------------------
def validate_medical_card(card: dict, drug_name: str):
    """
    Vérifie que la carte contient bien tous les champs requis.
    Corrige les types si nécessaire.
    """

    template = {
        "drug": drug_name,
        "class": "",
        "indications": [],
        "mechanism": "",
        "dosage": "",
        "side_effects": [],
        "contraindications": [],
        "interactions": []
    }

    for key, default in template.items():

        if key not in card:
            card[key] = default
            continue

        # Assurer les types corrects
        if isinstance(default, list) and not isinstance(card[key], list):
            card[key] = [card[key]] if card[key] else []

        if isinstance(default, str) and not isinstance(card[key], str):
            card[key] = str(card[key])

    return card


# ------------------------------------------------------
# 🧠 3. FONCTION PRINCIPALE : ULTRA-ROBUSTE
# ------------------------------------------------------
def generate_medical_card(drug: str):
    """
    Génère une carte médicale fiable, nettoyée et validée.
    """

    prompt = f"""
Tu es un expert médical. Ta mission : générer UNE SEULE fiche médicale fiable
pour le médicament suivant : "{drug}".

RÈGLES STRICTES :
- Tu réponds EXCLUSIVEMENT en JSON valide.
- Pas de texte avant ou après le JSON.
- Remplis seulement ce que tu connais avec certitude.
- Si tu n'es pas sûr, mets une chaîne vide "" ou une liste vide [].

FORMAT EXACT À RESPECTER :
{{
  "drug": "{drug}",
  "class": "",
  "indications": [],
  "mechanism": "",
  "dosage": "",
  "side_effects": [],
  "contraindications": [],
  "interactions": []
}}
"""

    try:
        response = client.chat.completions.create(
            model="hosted_vllm/Llama-3.1-70B-Instruct",
            messages=[
                {"role": "system", "content": "Assistant médical strict, factuel, sans spéculation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.15,   # 🔥 plus basse → moins d'hallucinations
            max_tokens=600
        )

        raw = response.choices[0].message.content.strip()

        # --- nettoyage JSON
        cleaned = clean_json_output(raw)

        try:
            parsed = json.loads(cleaned)
        except Exception:
            print("⚠ JSON invalide. Sortie brute :")
            print(raw)
            return {"drug": drug, "error": "Invalid JSON", "raw": raw}

        # Validation-type + complétion des champs
        validated = validate_medical_card(parsed, drug)

        return validated

    except Exception as e:
        print(f"⚠ Erreur génération carte médicale pour {drug}: {e}")
        return {"drug": drug, "error": str(e)}
    




# card = generate_medical_card("Paracetamol")
# print(json.dumps(card, indent=2, ensure_ascii=False))















# import json

def add_drug_to_dataset(drug_name, medical_card, df, dataset_path):
    # Convert JSON → string
    med_card_str = json.dumps(medical_card, ensure_ascii=False)

    new_row = {
        "canonical": drug_name,
        "med_card": med_card_str
    }

    df =  pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)  # ✅ Works 
    # df.append(new_row, ignore_index=True)

    # Save to disk
    df.to_csv(dataset_path, index=False, encoding="utf-8")

    return df


# HADI EL API mta3 el OCR + rag 

# # Charger dataset
# dataset_path = r"C:\Users\hp\Desktop\4ème\projet\medicaments_clean_for_ocr.csv"
# df = pd.read_csv(dataset_path)

# # Vérifier si la colonne existe sinon la créer
# if "med_card" not in df.columns:
#     df["med_card"] = ""  # colonne vide
#     df.to_csv(dataset_path, index=False, encoding="utf-8")
#     print("🆕 Colonne 'med_card' ajoutée au dataset.")
# else:
#     print("✔ Colonne 'med_card' déjà existante."





def process_drug(med, df, index, embeddings, dataset_path):

    # 1️⃣ Correction LLM
    corrected_name = correct_medicine_name_with_llama(
        med["drug"], med["dosage"], med["duration"]
    )
    print(f"\n🔧 Nom corrigé : {corrected_name}")

    # 2️⃣ Matching embedding sur NOM CORRIGÉ
    match, score = match_drug_embeddings(corrected_name, df, index, embeddings)

    # 3️⃣ Si médicament connu
    if match:
        print(f"✔ Match trouvé : {corrected_name} → {match} (score={score:.2f})")

        row = df[df["canonical"] == match].iloc[0]
        raw_card = row["med_card"]

        # ---- CAS : Carte déjà existante ----
        if isinstance(raw_card, str) and raw_card.strip() not in ["", "nan", "None"]:
            print(f"📄 Carte médicale trouvée pour {match}.")
            med_card = json.loads(raw_card)
            return match, med_card, df, index, embeddings

        # ---- CAS : Carte manquante → générer nouvelle carte ----
        print(f"⚠️ Carte médicale absente dans dataset pour {match}. Génération en cours…")
        med_card = generate_medical_card(match)

        # Mise à jour dataset
        df.loc[df["canonical"] == match, "med_card"] = json.dumps(med_card, ensure_ascii=False)
        df.to_csv(dataset_path, index=False, encoding="utf-8")

        return match, med_card, df, index, embeddings

    # 4️⃣ Aucun match → nouveau médicament
    print(f"❌ Aucun match pour {corrected_name} → génération carte médicale…")
    med_card = generate_medical_card(corrected_name)

    # Ajout dans dataset
    new_row = {
        "canonical": corrected_name,
        "med_card": json.dumps(med_card, ensure_ascii=False)
    }
    
    df =  pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)  # ✅ Works
    # df.append(new_row, ignore_index=True)
    df.to_csv(dataset_path, index=False, encoding="utf-8")

    print(f"🆕 Médicament ajouté au dataset : {corrected_name}")

    # Rebuild embeddings & FAISS
    embeddings = compute_embeddings(df["canonical"].tolist())
    index = create_faiss_index(embeddings)

    return corrected_name, med_card, df, index, embeddings



def pipeline(data_path,img_path):
    
    print("📌 Chargement du dataset...")
    df = pd.read_csv(data_path)

    # Ajouter colonne med_card si elle n'existe pas
    if "med_card" not in df.columns:
        df["med_card"] = ""
        df.to_csv(data_path, index=False, encoding="utf-8")
        print("🆕 Colonne 'med_card' ajoutée.")

    print("📌 Calcul des embeddings...")
    embeddings = compute_embeddings(df["canonical"].tolist())

    print("📌 Création de l'index FAISS...")
    index = create_faiss_index(embeddings)


    ocr_text = extract_text(img_path)
    
    print("\n🧪 OCR fourni :")
    print(ocr_text)


        
    extracted = extract_and_correct_meds_with_llama(ocr_text)

    print("\n🔍 Médicaments extraits (brut OCR) :")
    print(json.dumps(extracted, indent=2, ensure_ascii=False))

    
    final_output = []

    for med in extracted:
        drug_name, med_card, df, index, embeddings = process_drug(
            med, df, index, embeddings, data_path
        )

        final_output.append({
            "drug": drug_name,
            "dosage": med["dosage"],
            "duration": med["duration"],
            "card": med_card
        })

        
    print("\n\n🎉=== RÉSULTAT FINAL DU PIPELINE ===🎉")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))


    return final_output





# #######################################
# # 🔥 TEST COMPLET DU PIPELINE 🔥
# #######################################

# # Emplacement réel de ton dataset
# dataset_path = "C:/Users/hp/Desktop/4ème/projet/medicaments_clean_for_ocr.csv"

# print("📌 Chargement du dataset...")
# df = pd.read_csv(dataset_path)

# # Ajouter colonne med_card si elle n'existe pas
# if "med_card" not in df.columns:
#     df["med_card"] = ""
#     df.to_csv(dataset_path, index=False, encoding="utf-8")
#     print("🆕 Colonne 'med_card' ajoutée.")

# print("📌 Calcul des embeddings...")
# embeddings = compute_embeddings(df["canonical"].tolist())

# print("📌 Création de l'index FAISS...")
# index = create_faiss_index(embeddings)


# ########################################
# # 🔍 TEST OCR INPUT
# ########################################

# ocr_text = """
# Tab ENZOFLan 5mg x5day
# PaniD 40mg before meals
# Augmentin 1g x3day
# """

# print("\n🧪 OCR fourni :")
# print(ocr_text)


# ########################################
# # 🧪 EXTRACTION BRUTE
# ########################################

# extracted = extract_and_correct_meds_with_llama(ocr_text)

# print("\n🔍 Médicaments extraits (brut OCR) :")
# print(json.dumps(extracted, indent=2, ensure_ascii=False))


# ########################################
# # 🔥 TRAITEMENT DE CHAQUE MÉDICAMENT
# ########################################

# final_output = []

# for med in extracted:
#     drug_name, med_card, df, index, embeddings = process_drug(
#         med, df, index, embeddings, dataset_path
#     )

#     final_output.append({
#         "drug": drug_name,
#         "dosage": med["dosage"],
#         "duration": med["duration"],
#         "card": med_card
#     })


# ########################################
# # 🎉 RESULTAT FINAL
# ########################################

# print("\n\n🎉=== RÉSULTAT FINAL DU PIPELINE ===🎉")
# print(json.dumps(final_output, indent=2, ensure_ascii=False))

# print("\n📌 Vérification : dataset mis à jour → OK")
# print("📌 Embeddings recalculés → OK")
# print("📌 Index FAISS reconstruit → OK")

# print("\n🎯 TEST TERMINÉ — TON PIPELINE FONCTIONNE ✔")
