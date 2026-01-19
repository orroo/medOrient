# description.py
# ----------------------------------------------------
# Generates a safe, non-medical visual explanation
# from handcrafted facial symmetry features.
# Also allows sending general prompts to your hosted LLM.
# ----------------------------------------------------

import httpx
from openai import OpenAI

# -------------------------------
# Hosted LLM setup (TokenFactory)
# -------------------------------
def get_client():
    http_client = httpx.Client(verify=False)  # Disable SSL verification
    client = OpenAI(
        api_key="sk-34ef7724dde84849819a687a17406604",  # 👉 replace with your key
        base_url="https://tokenfactory.esprit.tn/api",
        http_client=http_client
    )
    return client

# -------------------------------
# 1️⃣ Explain facial symmetry features safely
def explain_features(features, severity=None):
    """
    features = {
        "smile_vertical_asymmetry": float,
        "mouth_horizontal_asymmetry": float,
        "eye_horizontal_asymmetry": float,
        "general_facial_symmetry": float
    }
    severity = float, optional severity value
    """

    note = ""
    if severity is not None and severity > 12:
        note = "\n⚠️ Note: The overall facial asymmetry is quite pronounced, so some visual differences may be noticeable."

    prompt = f"""
Vous recevrez des valeurs numériques décrivant la symétrie du visage.
Ces valeurs NE SONT PAS médicales et doivent être interprétées uniquement visuellement.

Tâche : expliquez en une phrase courte et concise (en français) à quel point
l'asymétrie faciale est visible. ci elle est commun ou pas , si elle est claire ou si elle est marquée. avec un peu de detaille 
N'utilisez pas les valeurs numériques dans laréponse, elles servent uniquement de référence.

Voici les valeurs (à ne PAS répéter) :
- Smile vertical asymmetry: {features['smile_vertical_asymmetry']}
- Mouth horizontal asymmetry: {features['mouth_horizontal_asymmetry']}
- Eye horizontal asymmetry: {features['eye_horizontal_asymmetry']}
- General facial symmetry: {features['general_facial_symmetry']}{note}

Directives :
- Classez la visibilité comme : "symétrie équilibrée", "asymétrie légère", "asymétrie perceptible" ou "asymétrie marquée".
- Décrivez très brièvement (1 phrase) ce qui est visible, par exemple « sourire légèrement inégal ».
- Terminez par une conclusion unique et concise.
- NE PAS évoquer de diagnostics médicaux.
- Répondez en FRANÇAIS, court et concis.

"""

    client = get_client()
    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content

# -------------------------------
# 2️⃣ General prompt function
# -------------------------------
def generate_description(prompt):
    """
    Send any text prompt to the hosted LLM and get the response.
    """
    client = get_client()
    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": "Tu es un assistant utile et concis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example features
    features_example = {
        "smile_vertical_asymmetry": 0.05,
        "mouth_horizontal_asymmetry": 0.03,
        "eye_horizontal_asymmetry": 0.01,
        "general_facial_symmetry": 0.04
    }

    print("[Visual Description]")
    print(explain_features(features_example))
