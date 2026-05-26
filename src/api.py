from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.agent import run_agent
import asyncio
import json

app = FastAPI()

# Αποθήκευση ιστορικού ανά session
session_histories = {}

# Ορισμός schemas
class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    history = session_histories.get(request.session_id, [])
    response, updated_history = run_agent(request.message, history)
    session_histories[request.session_id] = updated_history
    return ChatResponse(response=response)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    history = session_histories.get(request.session_id, [])
    response, updated_history = run_agent(request.message, history)
    session_histories[request.session_id] = updated_history

    async def generate():
        # Στέλνουμε token-by-token με SSE format
        for char in response:
            data = json.dumps({"token": char})
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.01)
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")