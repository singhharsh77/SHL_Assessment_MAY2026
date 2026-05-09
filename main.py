from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any
from agent import process_chat
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Recommender API")

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Pass messages to agent logic
    response_data = process_chat(request.messages)
    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
