from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_community.document_loaders import PyPDFLoader
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from pathlib import Path
from store import vectorstore
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from tempfile import mkdtemp
from langchain_docling.loader import ExportType
import bs4
from fastapi.responses import PlainTextResponse
import shutil
import os





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

EXPORT_TYPE = ExportType.DOC_CHUNKS
EMBED_MODEL_ID = "BAAI/bge-large-en-v1.5"
MILVUS_URI = str(Path(mkdtemp()) / "docling.db")



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

        global vectorstore 
        vectorstore = None
        FILE_PATH = temp_path
        loader = DoclingLoader(
        file_path=FILE_PATH,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
        )

        docs = loader.load()

        if EXPORT_TYPE == ExportType.DOC_CHUNKS:
            splits = docs
        elif EXPORT_TYPE == ExportType.MARKDOWN:
            from langchain_text_splitters import MarkdownHeaderTextSplitter

            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
                ],
            )
            splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
        else:
            raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")

        embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

        milvus_uri = str(Path(mkdtemp()) / "docling.db")  # or set as needed
        vectorstore = Milvus.from_documents(
            documents=splits,
            embedding=embedding,
            collection_name="docling_demo",
            connection_args={"uri": milvus_uri},
            index_params={"index_type": "FLAT"},
            drop_old=True,
        )
        return {"status": "success", "pages": len(docs)}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



@app.post("/chat",  response_class=PlainTextResponse)
async def chat(request: ChatRequest):
    from main import graph
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

    return response_text or "No response"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)