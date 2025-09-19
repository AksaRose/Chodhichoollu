# server.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from main import graph  # import graph from main.py
from typing import Optional, Dict, List
import uuid

app = FastAPI()

conversation_store: Dict[str, List[Dict]] = {}

class Query(BaseModel):
    question: str
    session_id: Optional[str] = None

class Response(BaseModel):
    response: str
    session_id: str

@app.post("/ask")
async def ask(query: Query) -> Response:

    session_id = query.session_id or str(uuid.uuid4())

    existing_messages = conversation_store.get(session_id, [])

    current_messages = existing_messages + [{"role": "user", "content": query.question}]
    
    state = {"messages": current_messages}
    
    final_state = graph.invoke(state)
    
    conversation_store[session_id] = final_state["messages"]
    
    return Response(
        response=final_state["messages"][-1].content,
        session_id=session_id
    )

@app.get("/conversations/{session_id}")
async def get_conversation(session_id: str):
    """Optional: Get conversation history"""
    messages = conversation_store.get(session_id, [])
    return {"session_id": session_id, "messages": messages}

@app.delete("/conversations/{session_id}")
async def clear_conversation(session_id: str):
    """Optional: Clear conversation history"""
    if session_id in conversation_store:
        del conversation_store[session_id]
        return {"message": f"Conversation {session_id} cleared"}
    return {"message": "Session not found"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)