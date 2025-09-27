from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from store import vector_store
from langchain_community.document_loaders import PyPDFLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
import os


from main import graph


app = FastAPI()

origins = [
    "http://localhost:5173",  
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"  


class ChatResponse(BaseModel):
    reply: str

class ChatUploadResponse(BaseModel):
    status: str
    pages: int = 0
    error: str = None


@app.post("/upload", response_model=ChatUploadResponse)
async def upload_file(file: UploadFile = File(...)):
        
    try:
        temp_path = f"./temp_{file.filename}"
        with open(temp_path,"wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        all_splits = text_splitter.split_documents(docs)

        vector_store.add_documents(all_splits)
        return {"status": "success", "pages": len(docs)}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    response_text = None
    for step in graph.stream(
        {"messages": [{"role": "user", "content": request.message}]},
        stream_mode="values",
        config=config,
    ):
        msg = step["messages"][-1]
        if msg.type == "ai":
            response_text = msg.content

    return ChatResponse(reply=response_text or "No response")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)