from fastapi import FastAPI, HTTPException
from orchestrator import load_dashboard

app = FastAPI(title="NBE Agentic Coach — Coordinator API")


@app.get("/dashboard/{customer_id}")
def get_dashboard(customer_id: str):
    result = load_dashboard(customer_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/")
def health_check():
    return {"status": "Coordinator is running"}
