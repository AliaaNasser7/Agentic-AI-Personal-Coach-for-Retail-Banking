"""
Chat Agent — reuses the existing specialist agents rather than duplicating
their logic. Routing is simple keyword-matching (not another LLM call) to
avoid stacking multiple slow local LLM calls into a single chat reply —
only ONE LLM call happens per message, for the final phrasing.
"""

import requests

from agent_stubs import run_spending_agent, run_goals_agent, run_simulation_agent

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"  # reuse the model already confirmed working end-to-end


def gather_context(customer_id: str, user_text: str) -> dict:
    """Keyword-based routing — decides which agent(s) to call based on the
    question, without needing an LLM to make that decision."""
    text = user_text.lower()
    context = {}

    if any(k in text for k in ["spend", "spending", "expense", "overspend"]):
        context["spending"] = run_spending_agent(customer_id)

    if any(k in text for k in ["goal", "saving", "save"]):
        context["goals"] = run_goals_agent(customer_id)

    if any(k in text for k in ["future", "project", "simulat", "forecast", "balance in", "balance", "months ahead", "next few months", "later"]):
        context["simulation"] = run_simulation_agent(customer_id)

    if not context:
        # Unclear question — pull the two lightest-weight agents so the LLM
        # has something to ground its answer in either way.
        context["spending"] = run_spending_agent(customer_id)
        context["goals"] = run_goals_agent(customer_id)

    return context


def build_chat_prompt(user_text: str, context: dict) -> str:
    context_lines = []

    if "spending" in context:
        s = context["spending"]
        if "error" not in s:
            context_lines.append(
                f"Spending data: avg monthly income {s['avg_monthly_income']}, "
                f"avg monthly spending {s['avg_monthly_spending']}, "
                f"monthly surplus {s['monthly_surplus']}."
            )

    if "goals" in context:
        g = context["goals"]
        if "error" not in g:
            for goal in g.get("goals", []):
                context_lines.append(
                    f"Goal '{goal['goal_name']}': target {goal['target_amount']}, "
                    f"status {goal['status']}, monthly gap {goal['monthly_gap']}."
                )

    if "simulation" in context:
        sim = context["simulation"]
        if "error" not in sim:
            bp = sim.get("baseline_projection", {})
            if bp.get("projection"):
                last = bp["projection"][-1]
                context_lines.append(
                    f"Balance projection: currently {bp.get('last_known_balance')}, "
                    f"projected to reach {last['projected_balance']} by {last['date']}."
                )

    context_block = "\n".join(
        context_lines) if context_lines else "No specific data available."

    return f"""You are a friendly personal finance coach for a retail bank customer.

Customer's question: "{user_text}"

Real data about this customer (use ONLY these numbers, never invent any):
{context_block}

STRICT RULES:
- Only use numbers that appear above. Never invent or estimate a number not given.
- If the data above doesn't answer the question, say so honestly instead of guessing.
- Keep your reply short (2-4 sentences), warm, and directly useful.
"""


def fallback_reply(context: dict) -> str:
    """Non-LLM safe reply, built directly from whatever data was gathered."""
    parts = []
    if "spending" in context and "error" not in context["spending"]:
        s = context["spending"]
        parts.append(
            f"Your average monthly surplus is {s['monthly_surplus']}.")
    if "goals" in context and "error" not in context["goals"]:
        for goal in context["goals"].get("goals", []):
            parts.append(
                f"Your '{goal['goal_name']}' goal is currently {goal['status']}.")
    if "simulation" in context and "error" not in context["simulation"]:
        bp = context["simulation"].get("baseline_projection", {})
        if bp.get("projection"):
            parts.append(
                f"Your projected balance in {len(bp['projection'])} months is {bp['projection'][-1]['projected_balance']}.")
    return " ".join(parts) if parts else "I couldn't find enough data to answer that right now."


def chat_with_coach(customer_id: str, user_text: str) -> dict:
    context = gather_context(customer_id, user_text)
    prompt = build_chat_prompt(user_text, context)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=150,
        )
        response.raise_for_status()
        reply = response.json()["message"]["content"].strip()
        used_fallback = False
    except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as e:
        print(f"[CHAT FALLBACK] LLM call failed: {e}")
        reply = fallback_reply(context)
        used_fallback = True

    return {
        "customer_id": customer_id,
        "reply": reply,
        "agents_used": list(context.keys()),
        "used_fallback": used_fallback,
    }
