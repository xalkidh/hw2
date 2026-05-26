from fastapi import FastAPI
from pydantic import BaseModel
from src.agent import run_agent

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
    # Παίρνουμε το ιστορικό του session
    history = session_histories.get(request.session_id, [])
    
    # Τρέχουμε τον agent
    response, updated_history = run_agent(request.message, history)
    
    # Αποθηκεύουμε το ενημερωμένο ιστορικό
    session_histories[request.session_id] = updated_history
    
    return ChatResponse(response=response)