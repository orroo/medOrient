from fastapi import FastAPI, Request ,HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_404_NOT_FOUND,HTTP_400_BAD_REQUEST
from pydantic import BaseModel 

from rag_func import *

app = FastAPI()

image_path = r"C:\Users\omarr\OneDrive\Desktop\New folder\ocr_api\WhatsApp Image 2025-12-02 à 15.34.03_a235721a.jpg"   # ou Desktop/3.jpeg si ton script est sur le bureau

data_path=r"C:\Users\omarr\OneDrive\Desktop\New folder\ocr_api\medicaments_clean_for_ocr.csv"


class QUESTION_Request(BaseModel):
    question: str

@app.post("/llm", response_class=HTMLResponse)
async def index(request: QUESTION_Request):
    """
    Render the main chat interface template with server and Grafana details.
    """
    try:
        final_output=pipeline(data_path,image_path)  #add your function here
        df = pd.read_csv(data_path)


        answer= chat_with_prescription(request.question, final_output, df)
        return JSONResponse(status_code=200, content={"answer":answer})

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"error: {str(e)}"
        )
    


# @app.get("/LLM", response_class=HTMLResponse)
# async def index(request: QUESTION_Request):
#     """
#     Render the main chat interface template with server and Grafana details.
#     """
#     try:
#         answer=pipeline(data_path,image_path)  #add your function here
        
#         return JSONResponse(status_code=200, content=answer)

#     except Exception as e:
#         print(e)
#         raise HTTPException(
#             status_code=HTTP_400_BAD_REQUEST,
#             detail=f"error: {str(e)}"
#         )
    





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