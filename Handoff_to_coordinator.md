# Spending Agent → Coordinator Handoff

**From:** [Your name] — Spending Agent
**To:** Person 1 — Architecture & Orchestrator Lead
**Status:** Ready to be wired into LangGraph 

---

## 1) The function you'll call

```python
from spending_agent_tools import analyze_customer, SpendingAgentError

try:
    result = analyze_customer(customer_id)
except SpendingAgentError as e:
    # invalid customer_id, missing data file, etc. — catch this exact exception type
    print(f"Spending Agent failed: {e}")
```

**You don't need to call anything else for the analysis** — this single function returns the full analysis, ready to drop into the shared State.

## 2) Output shape (JSON-ready, verified)

```json
{
  "customer_id": "CUST59072A",
  "avg_monthly_income": 27700.0,
  "avg_monthly_spending": 21956.4,
  "monthly_surplus": 5743.6,
  "months_analyzed": 6,
  "overspending_categories": [
    {
      "month": "2026-04",
      "category": "Shopping",
      "month_amount": 3927.21,
      "avg_other_months": 1273.03,
      "vs_avg_pct": 208.5
    }
  ],
  "analyzed_by": "Spending Agent"
}
```
If there's no overspending, `overspending_categories` returns `[]` (not null).

## 3) If you need a ready-made text message for the customer (not just numbers)

```python
from spending_agent_local import generate_coach_message

result = generate_coach_message(customer_id)
# {"customer_id": ..., "coach_message": "...", "guardrail_warning": None or str}
```
**Requires Ollama running locally** (`ollama serve`) — if it isn't running, this raises a `ConnectionError` with a clear message.

**Important:** if `guardrail_warning` is not `None`, that means something looked suspicious in the message (invented info, a real issue being ignored, or a factual contradiction). **Don't send that message to the customer as-is** — this case needs review or a fallback to a generic message instead of the flagged one.

## 4) Errors

Every failure is raised as a single `SpendingAgentError` type, not scattered exception types — just catch this one type and you'll cover all cases (invalid customer_id, empty, None, missing file...).

## 5) Things we need to align on

-  Will `customer_id` reach me in the same format as the current dataset (`CUSTXXXXXX`)?
-  Will the Coordinator call `analyze_customer` only, or `generate_coach_message` too (i.e. should the text message be generated on my end, or by Person 4 - Behavioral Agent)?
-  If `guardrail_warning` fires, do you want me to return something different than the original message, or will the Coordinator decide what to do with it?

## 6) Reference files

- `README.md` — full documentation of all functions + guardrails
- `demo_spending_agent.py` — a runnable walkthrough showing all scenarios
- `test_spending_agent.py` — 25 unit tests (`python -m unittest test_spending_agent.py -v`)