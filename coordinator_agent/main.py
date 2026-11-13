from fastapi import FastAPI, HTTPException
from coordinator_agent.agent_stubs import run_goals_agent, run_simulation_agent, run_spending_agent
from pydantic import BaseModel
from orchestrator import load_dashboard
from chat_agent import chat_with_coach

app = FastAPI(title="NBE Agentic Coach — Coordinator API")


class ChatMessage(BaseModel):
    text: str

@app.get("/dashboard/{customer_id}")
def get_dashboard(customer_id: str):
    result = load_dashboard(customer_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/chat/{customer_id}")
def chat(customer_id: str, message: ChatMessage):
    return chat_with_coach(customer_id, message.text)


@app.get("/")
def health_check():
    return {"status": "Coordinator is running"}
