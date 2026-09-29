"""
Optional coach-message layer on top of the Spending Agent's numeric
analysis. Ported from the team's `spending_agent_local.py`.

This is NOT used by the Recommendation Agent (the Recommendation Agent
works from raw numbers, per the coordinator's design decision — see
coordinator-agen_README.md: "The Overview tab's combined message is
generated separately by the Recommendation Agent from the raw numbers,
not from an already-summarized message").

This module only feeds the Spending tab's own text. Requires a local
Ollama server; if it isn't running, callers should catch ConnectionError
and fall back to a plain sentence built from the numbers (the orchestrator
already does this).
"""
import json
import requests

from agents.spending_agent import (
    calculate_monthly_surplus,
    detect_overspending_categories,
)
from data_access import load_customers

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"


def build_prompt(customer_id: str) -> str:
    customers = load_customers()
    customer_row = customers[customers["customer_id"] == customer_id].iloc[0]

    surplus_data = calculate_monthly_surplus(customer_id)
    overspending = detect_overspending_categories(customer_id, scope="any_month")
    overspending_summary = overspending[:3] if overspending else []

    overspending_block = (
        json.dumps(overspending_summary, indent=2)
        if overspending_summary
        else "NONE. Spending this period is consistent with the customer's usual habits. Do not mention any specific category or amount as a concern."
    )

    return f"""You are a friendly, encouraging personal finance coach for a retail bank customer.

Customer profile:
- Name: {customer_row['name']}
- Age: {customer_row['age']}
- Job: {customer_row['job']}
- Monthly income: {customer_row['monthly_income']} EGP

Financial analysis (calculated from their real transaction data):
- Average monthly income: {surplus_data['avg_monthly_income']} EGP
- Average monthly spending: {surplus_data['avg_monthly_spending']} EGP
- Monthly surplus: {surplus_data['monthly_surplus']} EGP

Overspending flags (categories where recent spending is notably above their own normal habit):
{overspending_block}

STRICT RULES:
- Only reference numbers, categories, and facts that appear explicitly above.
- If there are no overspending flags, do not name any category as a concern; praise the surplus and give one generic tip instead.
- If there are overspending flags, explicitly name at least the top one and its percentage.
- Never invent a number not given above.

Write a short coaching message (3-5 sentences), warm and human, ending with one concrete suggestion.
"""


def call_llm(prompt: str) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=body, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not reach the local Ollama server at http://localhost:11434.\n"
            "Install from https://ollama.com, run `ollama pull llama3.1:8b`, then `ollama serve`."
        )
    data = response.json()
    return data["message"]["content"].strip()


def generate_coach_message(customer_id: str) -> dict:
    """Returns {"customer_id", "coach_message", "guardrail_warning"}.
    Raises ConnectionError if Ollama isn't running — callers should catch
    this and fall back to a plain numeric sentence."""
    prompt = build_prompt(customer_id)
    message = call_llm(prompt)
    return {
        "customer_id": customer_id,
        "coach_message": message,
        "guardrail_warning": None,  # full guardrail checks live in the original branch file;
                                    # omitted here to keep this port minimal and focused.
    }
