from fastapi import FastAPI, Request ,HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_404_NOT_FOUND,HTTP_400_BAD_REQUEST
from pydantic import BaseModel 
from functions import *

app = FastAPI()



from tkinter.filedialog import askopenfilename

pdf_path = [r"C:\Users\omarr\OneDrive\Desktop\New folder\pdf_pi\zahrouni maya (1).pdf"]


class QUESTION_Request(BaseModel):
    question: str




@app.post("/llm", response_class=HTMLResponse)
async def index(request: QUESTION_Request):
    """
    Render the main chat interface template with server and Grafana details.
    """
    try:
        print("question",request.question)
        print("pdf_path",pdf_path)
        answer=analyse_pdf_chat(request.question, pdf_path, ESPRIT_API_KEY)
        return JSONResponse(content={
        "answer": answer
    })

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"error: {str(e)}"
        )
    



import uvicorn





app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)})

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8888)