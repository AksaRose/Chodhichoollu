from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from main import graph


app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"  


class ChatResponse(BaseModel):
    reply: str

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