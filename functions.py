
# -----------------------------
# 🔹 BM25 INDEX
# -----------------------------

from openai import OpenAI
import httpx
import json


import json
import textwrap

import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any , Optional
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dataclasses import dataclass, field
import time




from dotenv import load_dotenv

load_dotenv()

TOP_K = int(os.getenv("TOP_K", "10"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2") # embeddings


API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

HF_LLM_MODEL = os.getenv("HF_LLM_MODEL")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")





@dataclass
class DocChunk:
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)




class BM25Index:
    def __init__(self, docs: List[DocChunk]):
        self.corpus = [d.text for d in docs]
        tokenized = [c.split() for c in self.corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.metas = [d.meta for d in docs]

    def search(self, query: str, k: int = TOP_K):
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[-k:][::-1]
        return [
            {"text": self.corpus[i], "meta": self.metas[i], "score": float(scores[i])}
            for i in top_idx
        ]


class FaissIndex:
    def __init__(self, emb_model_name=EMBED_MODEL):
        print("[faiss] chargement modèle embeddings...")
        # self.model = SentenceTransformer(emb_model_name)
        try : 
            self.model = SentenceTransformer('models/'+emb_model_name)
            print( "loaded locally")
        except :
            # If not found, load the model from the internet and save it locally
            self.model= SentenceTransformer("sentence-transformers/"+emb_model_name)
            print("saving")
            self.model.save('models/'+emb_model_name)  # Save locally
            print("saved")

        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.embs = None

    def add(self, docs: List[DocChunk]):
        texts = [d.text for d in docs]
        print(f"[faiss] encodage de {len(texts)} chunks...")
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embs)
        self.index.add(embs)

        self.texts.extend(texts)
        self.metas.extend([d.meta for d in docs])
        self.embs = embs if self.embs is None else np.vstack([self.embs, embs])

    def search(self, query: str, k: int = TOP_K):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        return [
            {"text": self.texts[i], "meta": self.metas[i], "score": float(s)}
            for i, s in zip(I[0], D[0]) if 0 <= i < len(self.texts)
        ]

    # ✅ Save / Load pour FAISS
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/store.pkl", "wb") as f:
            pickle.dump({"texts": self.texts, "metas": self.metas, "embs": self.embs}, f)
        print(f"[faiss] index sauvegardé dans {path}")

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/store.pkl", "rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.metas = data["metas"]
        self.embs = data["embs"]
        print(f"[faiss] index chargé depuis {path}")

# -----------------------------
# 🔹 Hybrid search
# -----------------------------
def hybrid_search(query: str, faiss_idx: FaissIndex, bm25_idx: BM25Index, k: int = TOP_K):
    dense = faiss_idx.search(query, k=k)
    sparse = bm25_idx.search(query, k=k)
    seen = set()
    merged = []
    for r in dense + sparse:
        cid = r["meta"].get("chunk_id")
        if cid not in seen:
            merged.append(r)
            seen.add(cid)
        if len(merged) >= k:
            break
    return merged


http_client = httpx.Client(verify=False)
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
    http_client=httpx.Client(timeout=httpx.Timeout(120.0))
)

def llm_generate_api(
    prompt: str,
    model_name: str = HF_LLM_MODEL,
    max_tokens: int = 256,
    temperature: float = 0.2
) -> str:

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response.choices[0].message.content.strip()



def decide_mode_and_response(query: str, context: str) -> Dict[str, any]:
    """
    Génère le mode (SYMPTÔMES ou MALADIE) et les questions ou fiche maladie
    à partir de la requête utilisateur et du contexte.
    """
    prompt = SYSTEM_PROMPT + "\n\n" + (
        "DOCUMENTS RÉCUPÉRÉS (contexte) :\n" + context[:3000] + "\n\n"
    ) + (
        "Tâche:\nL'utilisateur a envoyé:\n" + query + "\n\n"
        "1) Indique en un mot le MODE choisi (SYMPTÔMES ou MALADIE).\n"
        "2) Si MODE=SYMPTÔMES : Génére 2 à 5 questions personnalisées et explique brièvement pourquoi.\n"
        "3) Si MODE=MALADIE : Fournis une fiche concise en français.\n\n"
        "⚠️ IMPORTANT: Répond strictement en JSON comme ceci:\n"
        '{"mode": "SYMPTÔMES", "data": {"questions": ["question1", ...]}}\n'
        "Ne mets aucun texte supplémentaire."
    )

    resp = llm_generate_api(prompt, model_name=HF_LLM_MODEL, max_tokens=600, temperature=0.0)
    # Debug
    print("DEBUG LLM raw response:", resp)

    # 🔹 Extraction JSON robuste
    try:
        jstart = resp.find("{")
        jend = resp.rfind("}")
        if jstart != -1 and jend != -1 and jend > jstart:
            jtxt = resp[jstart:jend + 1]
            parsed = json.loads(jtxt)
            mode = parsed.get("mode", "UNKNOWN").upper()
            data = parsed.get("data", {})
            if mode == "SYMPTÔMES" and "questions" not in data:
                data["questions"] = []
            return {"mode": mode, "data": data}
    except Exception as e:
        print("JSON parsing error:", e)

    # Fallback si parsing échoue
    return {"mode": "UNKNOWN", "data": {"questions": [], "raw": resp}}



def synthesize_from_passages_for_disease(passage_list: List[Dict[str, Any]], disease_name: str) -> str:
    """
    Génère une fiche complète pour une maladie à partir des passages RAG,
    en utilisant le LLM via API.
    """
    # 🔹 Construire le contexte à partir des passages récupérés
    context = "\n\n".join(
        [f"Source: {p['meta'].get('source','unknown')} p{p['meta'].get('page','?')}\n{p['text']}"
         for p in passage_list]
    )

    # 🔹 Construire le prompt
    prompt = (
        f"Vous êtes un médecin rédacteur. En vous basant strictement sur le contexte ci-dessous, "
        f"rédigez une fiche complète en français pour la maladie: '{disease_name}'. "
        "La fiche doit contenir : description, symptômes typiques, facteurs de risque, "
        "examens recommandés, prise en charge générale, spécialité à consulter.\n\n"
        f"Contexte (maximum 4000 caractères) :\n{context[:4000]}\n\n"
        "Fiche maladie :"
    )

    # 🔹 Appel LLM via API
    out = llm_generate_api(
        prompt,
        model_name=HF_LLM_MODEL,
        max_tokens=400,
        temperature=0.0
    )

        # Nettoyage minimal et ajout de la mention légale
    out = out.rstrip() + (
        "\n\n⚠️ Je ne suis pas un médecin, cette information est à titre informatif uniquement."
        "\nℹ️ Il est recommandé de consulter un spécialiste pour un avis médical précis."
    )

    # 🔹 Retourne le texte complet
    return out


def run_console_agent(faiss_idx: FaissIndex, bm25_idx: BM25Index):
    print("\n=== Agentic Medical RAG FR (API) ===")
    print("Disclaimer: Outil à visée informative uniquement.")
    patient_memory = {}

    while True:
        user = input("\nPatient (ou 'exit' pour quitter) : ").strip()
        if user.lower() in ("exit", "quit"):
            break

        passages = hybrid_search(user, faiss_idx, bm25_idx, k=TOP_K)
        context_text = "\n\n".join([p["text"] for p in passages])

        dec = decide_mode_and_response(user, context_text)
        mode = (dec.get("mode") or "").upper()
        data = dec.get("data", {})

        if mode == "SYMPTÔMES":
            questions = data.get("questions", [])
            if not questions:
                print("\n[Assistant] Aucun question générée par le modèle.")
                continue

            answers = {}
            print("\n[Assistant] Pour mieux comprendre, j'ai quelques questions :")
            for i, q in enumerate(questions, start=1):
                if not q:
                    continue
                a = input(f"[Q{i}] {q}\n> ").strip()
                answers[f"q{i}"] = {"q": q, "a": a}
                patient_memory[q] = a

            # Refaire une recherche enrichie avec les réponses
            enriched_query = user + " " + " ".join(v["a"] for v in answers.values())
            passages2 = hybrid_search(enriched_query, faiss_idx, bm25_idx, k=TOP_K)
            context2 = "\n\n".join(
                [f"Source:{p['meta']['source']} p{p['meta']['page']}\n{p['text']}" for p in passages2]
            )

            final_prompt = (
                SYSTEM_PROMPT + "\n\n"
                "Contexte médical récupéré:\n" + context2[:4000] + "\n\n"
                f"L'utilisateur a ces symptômes: {user}\n"
                f"Réponses utilisateurs aux questions: {json.dumps(answers, ensure_ascii=False)}\n\n"
                "1) Propose 1-3 hypothèses plausibles, en expliquant brièvement pour chaque pourquoi.\n"
                "2) Indique quel(s) examen(s) initial(aux) serait pertinent.\n"
                "3) Propose quelle spécialité consulter en priorité.\n"
            )
            final_out = llm_generate_api(final_prompt, model_name=HF_LLM_MODEL, max_tokens=500, temperature=0.0)
            print("\n[Assistant] Synthèse & hypothèses :\n")
            print(
    textwrap.indent(
        final_out.rstrip() + (
            "\n\n⚠️ Je ne suis pas un médecin, cette information est à titre informatif uniquement."
            "\nℹ️ Il est recommandé de consulter un spécialiste pour un avis médical précis."
        ),
        "  "
    )
)


        elif mode == "MALADIE":
            disease_name = user.strip()
            passages_d = hybrid_search(disease_name, faiss_idx, bm25_idx, k=TOP_K)
            fiche = synthesize_from_passages_for_disease(passages_d, disease_name)
            print("\n[Assistant] Fiche maladie :\n")
            print(textwrap.indent(fiche.strip(), "  "))

        else:
            print("\n[Assistant] Mode inconnu. Le modèle n'a pas pu déterminer SYMPTÔMES ou MALADIE.")


            #  LOAD INDEXES (fast)

# import pickle

# INDEX_DIR = ".\indexes"

# # Charger FAISS
# faiss_idx = FaissIndex(EMBED_MODEL)
# faiss_idx.load(f"{INDEX_DIR}/faiss")

# # Charger BM25

# with open(f"{INDEX_DIR}/bm25.pkl", "rb") as f:
#     bm25_idx = pickle.load(f)

# print("✅ Index chargés")



# # RUN CONSOLE

# run_console_agent(faiss_idx, bm25_idx)

