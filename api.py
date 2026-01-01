from fastapi import FastAPI, Request ,HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_404_NOT_FOUND,HTTP_400_BAD_REQUEST
from pydantic import BaseModel 


app = FastAPI()



class QUESTION_Request(BaseModel):
    question: str

@app.get("/llm", response_class=HTMLResponse)
async def index(request: QUESTION_Request):
    """
    Render the main chat interface template with server and Grafana details.
    """
    try:
        #### make sure bich tbaddel el system wo user prompt kima t7ib 
        system_prompt="""you are a medical chatbot .  
        you will be given some question and you will give an answer to it. 
        use the given context to answer the question.
        """
        #### #### #### #### #### #### #### #### #### #### #### #### #### 
        user_prompt=f"""question:
        {request.question} 
        """
        #### #### #### #### #### #### #### #### #### #### #### #### #### 

        answer=query_LLM(system_prompt,user_prompt)  #add your function here
        
        return{"answer":answer}

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