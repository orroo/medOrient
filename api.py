from fastapi import FastAPI, Request, HTTPException, Query 
from fastapi.responses import JSONResponse
import hmac
import hashlib
import uvicorn
import sys
import httpx
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from functions import *

import requests
import json

app = FastAPI()

class chat_Request(BaseModel):
    question: str




INDEX_DIR = os.getenv("INDEX_DIR")

# Charger FAISS
faiss_idx = FaissIndex(EMBED_MODEL)
faiss_idx.load(f"{INDEX_DIR}/faiss")

# Charger BM25

with open(f"{INDEX_DIR}/bm25.pkl", "rb") as f:
    bm25_idx = pickle.load(f)

print("✅ Index chargés")

patient_memory = {}

@app.post("/chatbot")
async def get_answer(request: chat_Request):
    try:
        user = request.question
        # if user.lower() in ("exit", "quit"):
        #     break

        passages = hybrid_search(user, faiss_idx, bm25_idx, k=TOP_K)
        context_text = "\n\n".join([p["text"] for p in passages])

        dec = decide_mode_and_response(user, context_text)
        mode = (dec.get("mode") or "").upper()
        data = dec.get("data", {})

        if mode == "SYMPTÔMES":
            # if questions:
            questions = data.get("questions", [])
            if not questions:
                print("\n[Assistant] Aucun question générée par le modèle.")
                return JSONResponse(status_code=500, content={"answer": "Aucun question générée par le modèle."})

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
            resp= "\n[Assistant] Fiche maladie :\n"
            resp = resp + textwrap.indent(fiche.strip(), "  ")
            return JSONResponse(status_code=200, content={"answer": resp})


        else:
            resp="\n[Assistant] Mode inconnu. Le modèle n'a pas pu déterminer SYMPTÔMES ou MALADIE."
            return JSONResponse(status_code=200, content={"answer": resp})

    except Exception as e:
        print(f"Error: {e}", file=sys.stdout, flush=True)
        return JSONResponse(status_code=500, content={"error": str(e)})





# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"Error: {exc}", file=sys.stdout, flush=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})

if __name__ == '__main__':
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=7575
    )