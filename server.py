# server.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from main import graph  # import graph from main.py

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    state = {"messages": [{"role": "user", "content": query.question}]}
    final_state = graph.invoke(state)
    return {"response": final_state["messages"][-1].content}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)