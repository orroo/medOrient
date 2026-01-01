# --- Import robuste pour LangChain Splitter ---
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("✔ Import depuis langchain.text_splitter")
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("✔ Import depuis langchain_text_splitters")
    except ImportError:
        try:
            from langchain_core.text_splitter import RecursiveCharacterTextSplitter
            print("✔ Import depuis langchain_core.text_splitter")
        except ImportError:
            raise ImportError(
                "❌ Impossible d'importer RecursiveCharacterTextSplitter.\n"
                "Installe une version récente :\n"
                "pip install -U langchain langchain-community langchain-core"
            )


import langchain
from langchain_core.prompts import ChatPromptTemplate
import httpx
from openai import OpenAI
import asyncio
import os

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



from dotenv import load_dotenv

load_dotenv()

ESPRIT_API_KEY=os.getenv("ESPRIT_API_KEY")

LLM_URL=os.getenv("LLM_URL")

EMBEDDING_MODEL_NAME=os.getenv("EMBEDDING_MODEL_NAME")


http_client = httpx.Client(verify=False)
client = OpenAI(
    api_key=ESPRIT_API_KEY,
    base_url=LLM_URL,
    http_client=http_client
)





def extract_text_from_pdfs(pdf_paths):
    """
    pdf_paths : liste de chemins PDF
    """
    text = ""
    for pdf in pdf_paths:
        with pdfplumber.open(pdf) as doc:
            for page in doc.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    return text.strip()


def create_vector_store(text):
    """
    Transforme le texte PDF en chunks + embeddings + FAISS retriever
    """

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=1000
    ).split_text(text)

    
    try : 
        embedding_model = HuggingFaceEmbeddings(model_name='models/'+EMBEDDING_MODEL_NAME)
        print( "loaded locally")
    except :
        # If not found, load the model from the internet and save it locally
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/"+EMBEDDING_MODEL_NAME
        )
        print("saving")
        embedding_model.save('models/'+EMBEDDING_MODEL_NAME)  # Save locally
        print("saved")

    return FAISS.from_texts(chunks, embedding_model)





def build_medical_prompt(question, context):
    template = f"""
Tu es un assistant médical intelligent.

RÔLE :
- Si le document contient des résultats médicaux : explique leur signification.
- Si c’est une ordonnance : identifie les médicaments et leur usage.
- Si c’est un rapport : résume les conclusions.
- Si le texte est administratif : explique les démarches.
- Si le texte est incomplet : donne la meilleure interprétation possible.

⚠️ RÈGLES :
- Pas de spéculation.
- Pas d’invention médicale.
- Utilise uniquement le CONTEXTE fourni.
- Si l’information manque → dire "Donnée absente du document".

📘 CONTEXTE :
{context}

❓ QUESTION :
{question}

💬 RÉPONSE :
"""

    # prompt = ChatPromptTemplate.from_template(template)
    # formatted = prompt.format(question=question, context=context)
    # if isinstance(formatted, dict):
    #     formatted = str(formatted)
    return template
# formatted


def call_esprit_llm(prompt, esprit_api_key):
   

    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": "Tu es un assistant médical et administratif utile et concis."},
            {"role": "user", "content": str(prompt)}
        ],
        temperature=0.4,
        max_tokens=700,
        top_p=0.9
    )

    return response.choices[0].message.content


def analyse_pdf_chat(question, pdf_paths, esprit_api_key):
    # Extraction du texte PDF
    text = extract_text_from_pdfs(pdf_paths)
    if not text:
        return "❌ Aucun texte détecté dans les PDF."

    print("passing extraction")

    # Création du store FAISS
    vector_store = create_vector_store(text)
    print("passing embedding")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print("passing retriever")

    docs = retriever.invoke(question)
    print("passing invoker")

    if not docs:
        return "❌ Aucun passage pertinent trouvé."

    # Context
    context = "\n\n".join([d.page_content for d in docs])

    print("passing LLM")

    # Prompt intelligent
    prompt = build_medical_prompt(question, str(context))
    print("calling LLM")
    # Appel API
    print(type(prompt))
    print(prompt)

    return call_esprit_llm(prompt, esprit_api_key)







# --- Extraire texte depuis un PDF ---
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


# --- Construire le vector store ---
def create_vector_store_from_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300
    )
    
    chunks = splitter.split_text(text)
    
    try : 
        embedding_model = HuggingFaceEmbeddings(model_name='models/'+EMBEDDING_MODEL_NAME)
        print( "loaded locally")
    except :
        # If not found, load the model from the internet and save it locally
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/"+EMBEDDING_MODEL_NAME
        )
        print("saving")
        embedding_model.save('models/'+EMBEDDING_MODEL_NAME)  # Save locally
        print("saved")

    
    vector_store = FAISS.from_texts(chunks, embedding_model)
    return vector_store




def build_medical_analysis_prompt(question, context):

    return f"""
Tu es un assistant médical ultra-fiable spécialisé dans l’analyse de documents biologiques.

Tu dois répondre STRICTEMENT à partir du texte fourni.
Aucune connaissance extérieure n’est autorisée.

────────────────────────────────────
📄 CONTEXTE (Données du document)
────────────────────────────────────
{context}

────────────────────────────────────
⚠️ RÈGLES ANTI-HALLUCINATION
────────────────────────────────────
1. Ne JAMAIS interpréter sans comparer aux valeurs normales présentes dans le document.
2. Ne JAMAIS écrire des phrases comme : “peut indiquer”, “probablement”, “infection”, “inflammation”.
3. Ne PAS inventer de diagnostic ou de maladie.
4. Si une information n'est pas dans le document → écrire : “Non indiqué dans le document.”
5. Réponse courte, factuelle et maximum 8 lignes.
6. Ne pas réécrire tout le document.

────────────────────────────────────
❓ QUESTION UTILISATEUR
{question}

────────────────────────────────────
🧪 RÉPONSE FACTUELLE (basée EXCLUSIVEMENT sur le document) :
"""



def ask_llm(prompt):
    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": "Assistant médical strictement factuel."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.0
    )
    return response.choices[0].message.content
