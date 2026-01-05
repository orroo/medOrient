from fastapi import FastAPI, Request, HTTPException, Query ,Depends 
from fastapi.responses import JSONResponse
import hmac
import hashlib
import uvicorn
import sys
import httpx
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from functions_img import *

import requests
import json
import base64
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



# response = run_kidney_image_query(
#     img_path="/content/sample_data/cancer/Tumor- (43).jpg",
#     encoder_model=encoder,
#     classifier_model=classifier,
#     label_encoder=le,
#     faiss_idx=faiss_idx,
#     bm25_idx=bm25_idx
# )

# print(response)


# image_path = "C:/Users/user/Desktop/4IA/archive/IMG_CLASSES/bcc/ISIC_0026988.jpg"
# pred_class, confidence = predict_and_show_image(
#     image_path=image_path,
#     model=model,
#     transform=transform,
#     class_names=class_names,
#     device=device
# )


patient_memory = {}
# Request/Response Models
class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    answers: Optional[Dict[str, str]] = None  # For follow-up answers

class SymptomResponse(BaseModel):
    mode: str
    questions: List[str]
    conversation_id: str
    message: str

class DiagnosisResponse(BaseModel):
    mode: str
    hypotheses: str
    synthesis: str
    disclaimer: str

class DiseaseResponse(BaseModel):
    mode: str
    disease_info: str

class FileRequest(BaseModel):
    """
    Request model for storing a file.
    """
    message:str
    file_name: str
    file_content: bytes



@app.post("/upload_file")
async def upload_file_endpoint(file_request: FileRequest):
    """
    Endpoint that stores a file sent through a POST request.
    """
    try:
        print(file_request.message)
        if (file_request.message.lower().find("rein") != -1) or (file_request.message.lower().find("kidney") != -1):
            file_path = os.path.join("./uploads/kidney/uploaded_files/", file_request.file_name)
            file_bytes = base64.b64decode(file_request.file_content)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            
            response = run_kidney_image_query(
                img_path=file_path,
                encoder_model=encoder,
                classifier_model=classifier,    
                label_encoder=le,
                faiss_idx=faiss_idx,
                bm25_idx=bm25_idx
            )
            print(response) 
        else : 
            file_path = os.path.join("./uploads/peau/uploaded_files/", file_request.file_name)
            file_bytes = base64.b64decode(file_request.file_content)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            
            response = predict_and_show_image(
                image_path=file_path,
                model=loaded_model,
                transform=loaded_transform,
                class_names=loaded_classes,
                device="cpu"
            )


            print(response) 
        return JSONResponse(content={ 
                "answer": response
            })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    # return JSONResponse(content={"message": "File uploaded successfully"}, status_code=201)





# @app.post("/store_file")
# async def store_file_endpoint(file_request: FileRequest):
#     """
#     Endpoint that stores a file and runs a function with the file path.
#     """
#     file_path = os.path.join("/uploads/uploaded_files/", file_request.file_name)
#     with open(file_path, "wb") as f:
#         f.write(file_request.file_content)
#     response = run_kidney_image_query(
#         img_path=file_path,
#         encoder_model=encoder,
#         classifier_model=classifier,
#         label_encoder=le,
#         faiss_idx=faiss_idx,
#         bm25_idx=bm25_idx
#     )
#     print(response) 
#     # result = run_function_with_file_path(file_path)
#     return JSONResponse(content={ 
#             "answer": response
#         })



# Main API endpoint
@app.post("/chatbot")
async def chat_endpoint(chat_request: ChatRequest):
    """
    API-focused medical chat endpoint that handles symptoms and disease queries.
    
    Flow:
    1. Initial request -> Returns questions if SYMPTÔMES mode
    2. Follow-up request with answers -> Returns diagnosis
    3. MALADIE mode -> Returns disease information directly
    """
    try:
        user_query = chat_request.question.strip()
        
        if not user_query:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Step 1: Initial search and mode detection
        passages = hybrid_search(user_query, faiss_idx, bm25_idx, k=TOP_K)
        context_text = "\n\n".join([p["text"] for p in passages])

        dec = decide_mode_and_response(user_query, context_text)
        mode = (dec.get("mode") or "").upper()
        data = dec.get("data", {})

        # Step 2: Handle based on mode
        if mode == "SYMPTÔMES":
            return await handle_symptoms_mode(
                user_query, 
                data, 
                chat_request.answers,
                chat_request.conversation_id
            )
        
        elif mode == "MALADIE":
            return await handle_disease_mode(user_query)
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unable to determine query type (SYMPTÔMES or MALADIE)"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


async def handle_symptoms_mode(
    user_query: str, 
    data: dict, 
    answers: Optional[Dict[str, str]],
    conversation_id: Optional[str]
) -> JSONResponse:
    """
    Handle symptom-based queries with a two-step process:
    1. First call: Return questions to ask user
    2. Second call: Process answers and return diagnosis
    """
    
    # First call: Generate questions
    if not answers:
        questions = data.get("questions", [])
        
        if not questions:
            raise HTTPException(
                status_code=500, 
                detail="No questions generated by the model"
            )
        
        # Generate unique conversation ID for tracking
        conv_id = conversation_id or generate_conversation_id()
        
        # Store initial query in session/cache for follow-up
        store_conversation_context(conv_id, {
            "query": user_query,
            "questions": questions
        })
        
        return JSONResponse(content={
            "mode": "SYMPTÔMES",
            "step": "questions",
            "questions": questions,
            "conversation_id": conv_id,
            "answer": "Pour mieux comprendre , j'ai des questions"
        })
    
    # Second call: Process answers and generate diagnosis
    else:
        # Retrieve original context
        context = get_conversation_context(conversation_id)
        if not context:
            raise HTTPException(
                status_code=400, 
                detail="Invalid or expired conversation_id"
            )
        
        original_query = context["query"]
        questions_asked = context["questions"]
        
        # Build enriched query with answers
        answer_text = " ".join(answers.values())
        enriched_query = f"{original_query} {answer_text}"

        print(enriched_query)
        
        # Enhanced search with answers
        passages = hybrid_search(enriched_query, faiss_idx, bm25_idx, k=TOP_K)
        context_text = "\n\n".join([
            f"Source: {p['meta']['source']} p{p['meta']['page']}\n{p['text']}" 
            for p in passages
        ])
        
        # Generate final diagnosis
        final_prompt = build_diagnosis_prompt(
            original_query, 
            context_text[:4000], 
            questions_asked,
            answers
        )
        
        diagnosis = llm_generate_api(
            final_prompt, 
            model_name=HF_LLM_MODEL, 
            max_tokens=500, 
            temperature=0.0
        )
        
        # Clean up conversation context
        clear_conversation_context(conversation_id)
        
        answer=f""" {diagnosis.strip()} ⚠️ This is not a medical diagnosis. This information is for informational purposes only.\n
                ℹ️ Please consult a healthcare professional for accurate medical advice."""

        return JSONResponse(content={
            "mode": "SYMPTÔMES",
            "step": "diagnosis",
            "synthesis": diagnosis.strip(),
            "disclaimer": (
                "⚠️ This is not a medical diagnosis. This information is for informational purposes only.\n"
                "ℹ️ Please consult a healthcare professional for accurate medical advice."
            ),
            "answer": answer
        })


async def handle_disease_mode(disease_name: str) -> JSONResponse:
    """
    Handle disease information queries - single-step process
    """
    passages = hybrid_search(disease_name, faiss_idx, bm25_idx, k=TOP_K)
    disease_info = synthesize_from_passages_for_disease(passages, disease_name)
    answer = f""" {disease_info.strip()} ⚠️ This is not a medical diagnosis. This information is for informational purposes only.\n"""
    return JSONResponse(content={
        "mode": "MALADIE",
        "answer": answer,
        "disease_name": disease_name,
        "information": disease_info.strip()
    })


def build_diagnosis_prompt(
    original_query: str, 
    context: str, 
    questions: List[str],
    answers: Dict[str, str]
) -> str:
    """Build the final diagnosis prompt"""
    
    qa_pairs = "\n".join([
        f"Q: {q}\nA: {answers.get(f'q{i+1}', 'No answer')}"
        for i, q in enumerate(questions)
    ])
    
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Medical context retrieved:\n{context}\n\n"
        f"User's initial symptoms: {original_query}\n\n"
        f"Questions and Answers:\n{qa_pairs}\n\n"
        "Based on this information:\n"
        "1) Propose 1-3 plausible hypotheses, briefly explaining each\n"
        "2) Indicate which initial examination(s) would be relevant\n"
        "3) Suggest which specialty to consult as a priority\n"
    )


# Helper functions (implement based on your storage solution)
def generate_conversation_id() -> str:
    """Generate unique conversation ID"""
    import uuid
    return str(uuid.uuid4())

def store_conversation_context(conv_id: str, context: dict):
    """Store conversation context (use Redis, database, or in-memory cache)"""
    # Example with in-memory cache (use Redis in production)
    conversation_cache[conv_id] = {
        "data": context,
        "timestamp": time.time()
    }

def get_conversation_context(conv_id: str) -> Optional[dict]:
    """Retrieve conversation context"""
    if not conv_id:
        return None
    cached = conversation_cache.get(conv_id)
    if cached:
        # Check expiration (e.g., 30 minutes)
        if time.time() - cached["timestamp"] < 1800:
            return cached["data"]
    return None

def clear_conversation_context(conv_id: str):
    """Clear conversation context after completion"""
    conversation_cache.pop(conv_id, None)


# In-memory cache (use Redis in production)
conversation_cache = {}




# @app.post("/chatbot")
# async def get_answer(request: chat_Request):
#     try:
#         user = request.question
#         # if user.lower() in ("exit", "quit"):
#         #     break

#         passages = hybrid_search(user, faiss_idx, bm25_idx, k=TOP_K)
#         context_text = "\n\n".join([p["text"] for p in passages])

#         dec = decide_mode_and_response(user, context_text)
#         mode = (dec.get("mode") or "").upper()
#         data = dec.get("data", {})

#         if mode == "SYMPTÔMES":
#             # if questions:
#             questions = data.get("questions", [])
#             if not questions:
#                 print("\n[Assistant] Aucun question générée par le modèle.")
#                 return JSONResponse(status_code=500, content={"answer": "Aucun question générée par le modèle."})

#             answers = {}
#             print("\n[Assistant] Pour mieux comprendre, j'ai quelques questions :")
#             for i, q in enumerate(questions, start=1):
#                 if not q:
#                     continue
#                 a = input(f"[Q{i}] {q}\n> ").strip()
#                 answers[f"q{i}"] = {"q": q, "a": a}
#                 patient_memory[q] = a

#             # Refaire une recherche enrichie avec les réponses
#             enriched_query = user + " " + " ".join(v["a"] for v in answers.values())
#             passages2 = hybrid_search(enriched_query, faiss_idx, bm25_idx, k=TOP_K)
#             context2 = "\n\n".join(
#                 [f"Source:{p['meta']['source']} p{p['meta']['page']}\n{p['text']}" for p in passages2]
#             )

#             final_prompt = (
#                 SYSTEM_PROMPT + "\n\n"
#                 "Contexte médical récupéré:\n" + context2[:4000] + "\n\n"
#                 f"L'utilisateur a ces symptômes: {user}\n"
#                 f"Réponses utilisateurs aux questions: {json.dumps(answers, ensure_ascii=False)}\n\n"
#                 "1) Propose 1-3 hypothèses plausibles, en expliquant brièvement pour chaque pourquoi.\n"
#                 "2) Indique quel(s) examen(s) initial(aux) serait pertinent.\n"
#                 "3) Propose quelle spécialité consulter en priorité.\n"
#             )
#             final_out = llm_generate_api(final_prompt, model_name=HF_LLM_MODEL, max_tokens=500, temperature=0.0)
#             print("\n[Assistant] Synthèse & hypothèses :\n")
#             print(
#     textwrap.indent(
#         final_out.rstrip() + (
#             "\n\n⚠️ Je ne suis pas un médecin, cette information est à titre informatif uniquement."
#             "\nℹ️ Il est recommandé de consulter un spécialiste pour un avis médical précis."
#         ),
#         "  "
#     )
# )


#         elif mode == "MALADIE":
#             disease_name = user.strip()
#             passages_d = hybrid_search(disease_name, faiss_idx, bm25_idx, k=TOP_K)
#             fiche = synthesize_from_passages_for_disease(passages_d, disease_name)
#             resp= "\n[Assistant] Fiche maladie :\n"
#             resp = resp + textwrap.indent(fiche.strip(), "  ")
#             return JSONResponse(status_code=200, content={"answer": resp})


#         else:
#             resp="\n[Assistant] Mode inconnu. Le modèle n'a pas pu déterminer SYMPTÔMES ou MALADIE."
#             return JSONResponse(status_code=200, content={"answer": resp})

#     except Exception as e:
#         print(f"Error: {e}", file=sys.stdout, flush=True)
#         return JSONResponse(status_code=500, content={"error": str(e)})





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