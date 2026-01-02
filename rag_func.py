from function import *



conversation_history = []  # mémoire du chat




def get_medical_context(drug_list, df):
    context_parts = []

    for drug in drug_list:
        # 🔹 match insensible à la casse
        mask = df["canonical"].astype(str).str.lower() == drug.lower()
        rows = df[mask]

        if rows.empty:
            print(f"⚠️ Aucun médoc trouvé dans le dataset pour : {drug}")
            continue

        row = rows.iloc[0]
        raw_card = row.get("med_card", "")

        # si pas de carte → on peut soit sauter, soit mettre un placeholder
        if pd.isna(raw_card) or raw_card in ["", "nan", None]:
            print(f"⚠️ med_card vide pour : {drug}")
            continue

        try:
            card = json.loads(raw_card)
        except Exception as e:
            print(f"⚠️ Erreur JSON pour med_card de {drug} : {e}")
            continue

        context_parts.append(json.dumps(card, ensure_ascii=False, indent=2))

    return "\n\n".join(context_parts)




def build_rag_prompt(question, context):
    return f"""
Tu es un expert médical spécialisé dans les interactions médicamenteuses.

Contexte clinique provenant des cartes médicales extraites du dataset :

{context}

Règles :
- Utilise PRIORITAIREMENT ce contexte pour répondre.
- Tu peux compléter avec tes connaissances médicales internes si nécessaire.
- Répond toujours de manière claire, simple et exacte.
- Mentionne explicitement les interactions possibles.
- Donne éventuellement des recommandations pratiques.

Question :
{question}

Réponse :
"""



def ask_medical_question(question, drug_list, df):
    context = get_medical_context(drug_list, df)
    prompt = build_rag_prompt(question, context)

    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": "Assistant médical fiable et prudent."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()





def ask_about_prescription(question, final_output, df):
    # Liste des médicaments reconnus après correction + matching
    drugs = [item["drug"] for item in final_output]

    print("\n📌 Médicaments concernés par la question :", drugs)

    answer = ask_medical_question(question, drugs, df)

    return answer







def ask_medical_question_conversational(question, drug_list, df):
    # 1. Construire contexte médical
    context = get_medical_context(drug_list, df)

    # 2. Construire le message utilisateur pour ce tour
    user_message = f"""
Contexte des médicaments :
{context}

Question de l'utilisateur :
{question}
"""

    # 3. Construire la conversation complète
    messages = [{"role": "system", "content": "Assistant médical expert, prudent, clair et fiable."}]

    # Ajouter historique
    for entry in conversation_history:
        messages.append({"role": entry["role"], "content": entry["content"]})

    # Ajouter le nouveau message
    messages.append({"role": "user", "content": user_message})

    # 4. Appeler le modèle
    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=messages,
        temperature=0.25,
        max_tokens=500
    )

    answer = response.choices[0].message.content.strip()

    # 5. Ajouter ce tour dans l'historique
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    return answer





def chat_with_prescription(question, final_output, df):
    drugs = [item["drug"] for item in final_output]
    answer = ask_medical_question_conversational(question, drugs, df)
    return answer
