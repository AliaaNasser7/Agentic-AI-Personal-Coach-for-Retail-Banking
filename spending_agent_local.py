"""
Setup :
    1. Install Ollama: https://ollama.com
    2. Download a model: ollama pull llama3.1:8b
"""

import json

import requests

from spending_agent_tools import (
    _load_customers,
    calculate_monthly_surplus,
    detect_overspending_categories,
)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"  

#Building the prompt 
def build_prompt(customer_id: str) -> str:
    customers = _load_customers()
    customer_row = customers[customers["customer_id"] == customer_id].iloc[0]

    surplus_data = calculate_monthly_surplus(customer_id)
    overspending = detect_overspending_categories(customer_id, scope="any_month")
    overspending_summary = overspending[:3] if overspending else []
    
    overspending_block = (
        json.dumps(overspending_summary, indent=2)
        if overspending_summary
        else "NONE. Spending this period is consistent with the customer's usual habits. Do not mention any specific category or amount as a concern."
    )

    prompt = f"""You are a friendly, encouraging personal finance coach for a retail bank customer.

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

STRICT RULES — follow these exactly:
- You may ONLY reference numbers, categories, and facts that appear explicitly above.
- If the overspending flags say NONE, you must NOT name any specific spending category
(no "dining", "shopping", etc.) as a concern, and you must NOT invent any amount.
In that case, praise the healthy surplus and give ONE generic, category-agnostic tip
(e.g. about growing savings or building an emergency fund) instead.
- If the overspending flags list ONE OR MORE categories, you MUST explicitly name at
least the top one and its percentage in your message. You must NOT say the customer's
spending is "consistent with their usual habits", "normal", or "on track" when real
overspending flags are listed above — that would directly contradict the data.
- Never state a number that was not given to you above.
- If you are unsure whether a detail is supported by the data above, leave it out.

Write a short coaching message (3-5 sentences) directly to the customer:
1. Start with something positive or neutral (don't lead with criticism).
2. If there ARE overspending flags above, mention the most significant one specifically,
using the real numbers, in a supportive (not judgmental) tone.
3. End with one concrete, encouraging suggestion.
4. Keep the tone warm and human — like a coach, not a bank statement.
"""
    return prompt


# Calling the model via Ollama
def call_llm(prompt: str) -> str:
    #Sends the prompt to a locally running Ollama server.
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,  # get the full response at once instead of token-by-token
    }

    try:
        response = requests.post(OLLAMA_URL, json=body, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not reach the local Ollama server at http://localhost:11434.\n"
            "Make sure Ollama is installed and running:\n"
            "  1. Install from https://ollama.com\n"
            "  2. Run: ollama pull llama3.1:8b\n"
            "  3. Run: ollama serve  (or just open the Ollama app)"
        )

    data = response.json()
    return data["message"]["content"].strip()


#Lightweight guardrail check
KNOWN_CATEGORIES = [
    "Rent/Installment", "Utilities", "Groceries", "Transportation",
    "Dining/Restaurants", "Entertainment", "Shopping", "Subscriptions",
    "Healthcare", "Savings Transfer",
]


def check_for_hallucinated_categories(message: str, overspending_summary: list) -> list:
    """
    Cheap sanity check: if the LLM's message names a spending category
    that was NOT in the actual overspending flags we gave it, that's a
    red flag the model may have invented a concern. Returns a list of
    suspicious category names found (empty list = looks clean).
    This is NOT a full guardrail agent , just a fast local check useful
    during development. A production system should validate more rules.
    """
    flagged_categories = {item["category"] for item in overspending_summary}
    suspicious = []

    message_lower = message.lower()
    for category in KNOWN_CATEGORIES:
        # crude keyword match (e.g. "Dining/Restaurants" -> "dining")
        keyword = category.split("/")[0].lower()
        if keyword in message_lower and category not in flagged_categories:
            suspicious.append(category)
    return suspicious

"""
# Phrases that claim spending is normal/consistent used to catch the model falsely reassuring the customer when a real flag was ignored.
_REASSURANCE_PHRASES = [
    "consistent with your usual", "consistent with their usual",
    "in line with your usual", "normal spending", "usual habits",
    "spending is consistent", "everything looks good", "on track",
]"""


def check_for_ignored_real_flags(message: str, overspending_summary: list) -> list:
    """
    the more dangerous direction: if there are real overspending flags,
    but the message uses reassuring language without naming any of the
    actually flagged categories, the model likely ignored a real
    warning. Returns the list of real flagged categories that were
    silently dropped (empty list = looks clean, or there was nothing to
    flag in the first place).
    """
    if not overspending_summary:
        return []  # nothing real was flagged, so nothing can be "ignored"

    message_lower = message.lower()
    flagged_categories = {item["category"] for item in overspending_summary}

    mentioned = {
        category for category in flagged_categories
        if category.split("/")[0].lower() in message_lower
    }

    if not mentioned:
        return sorted(flagged_categories)

    return []


def check_for_surplus_deficit_contradiction(message: str, monthly_surplus: float):
    """
    Returns a warning string if the message uses financially contradictory
    language relative to the customer's real monthly_surplus value, else None.
    """
    message_lower = message.lower()

    if monthly_surplus > 0 and "deficit" in message_lower:
        return (f"Message uses the word 'deficit' but the customer actually has "
                f"a positive surplus of {monthly_surplus} EGP.")

    if monthly_surplus < 0 and "surplus" in message_lower and "deficit" not in message_lower:
        return (f"Message implies a surplus but the customer actually has "
                f"a deficit of {monthly_surplus} EGP.")

    return None




# Full pipeline
def generate_coach_message(customer_id: str) -> dict:
    prompt = build_prompt(customer_id)
    message = call_llm(prompt)

    surplus_data = calculate_monthly_surplus(customer_id)
    overspending = detect_overspending_categories(customer_id, scope="any_month")
    overspending_summary = overspending[:3] if overspending else []

    hallucinated = check_for_hallucinated_categories(message, overspending_summary)
    ignored = check_for_ignored_real_flags(message, overspending_summary)
    contradiction = check_for_surplus_deficit_contradiction(message, surplus_data["monthly_surplus"])

    warnings = []
    if hallucinated:
        warnings.append(f"Message mentions categories not present in the actual data: {hallucinated}")
    if ignored:
        warnings.append(
            f"Message reassures the customer everything is normal but IGNORES real "
            f"overspending flags: {ignored}"
        )
    if contradiction:
        warnings.append(contradiction)

    return {
        "customer_id": customer_id,
        "coach_message": message,
        "guardrail_warning": " | ".join(warnings) if warnings else None,
    }

#manual test
if __name__ == "__main__":
    customers = _load_customers()
    sample_id = customers.iloc[0]["customer_id"]

    print(f"Generating coach message for customer: {sample_id} (local model via Ollama)\n")

    print("Prompt sent to the model")
    print(build_prompt(sample_id))

    print("\nCoach message")
    try:
        result = generate_coach_message(sample_id)
        print(result["coach_message"])
        if result["guardrail_warning"]:
            print(f"\n[GUARDRAIL WARNING] {result['guardrail_warning']}")
    except ConnectionError as e:
        print(f"[Skipped LLM call] {e}")