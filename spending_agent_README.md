# Spending Agent — README

## What this is
The Spending Agent for the NBE Agentic AI Personal Coach. It reads a
customer's transaction history and produces:
1. Deterministic numeric analysis (surplus, overspending flags) — no LLM.
2. A natural-language coach message built from that analysis — via a
   local LLM (Ollama), with **3 lightweight guardrail checks** against
   common LLM failure modes (see below).

## Files
| File | Purpose |
|---|---|
| `spending_agent_tools.py` | Core analysis functions (no LLM). Depends on `customers.csv` + `transactions.csv` in the same folder. |
| `spending_agent_local.py` | Turns the analysis into a human coach message, using a local Ollama model + guardrails. Depends on `spending_agent_tools.py`. |
| `demo_spending_agent.py` | Presentation-friendly walkthrough across 3 customer scenarios (normal / real anomaly / lowest income). |
| `test_spending_agent.py` | 25 unit tests (`python -m unittest test_spending_agent.py -v`). |
| `customers.csv`, `transactions.csv` | Synthetic data (from `generate_all_data.py`). |

> Note: some earlier drafts of this agent used the filename
> `spending_agent_llm_local.py` — the team's working copy is
> `spending_agent_local.py`. Same content, just confirm which name is
> actually imported wherever this agent is wired in.

## Setup
1. `pip install pandas requests`
2. Install [Ollama](https://ollama.com) and pull a model:
   ```
   ollama pull llama3.1:8b
   ```
   (if your machine runs out of memory, try a smaller model like `llama3.2:3b`)
3. Make sure `customers.csv` and `transactions.csv` are in the same folder
   as the scripts.

## How the Coordinator should call this agent

```python
from spending_agent_tools import analyze_customer, SpendingAgentError

try:
    result = analyze_customer(customer_id)
    # result is a plain, JSON-serializable dict — safe to drop straight
    # into the shared State under `financial_analysis`.
except SpendingAgentError as e:
    # customer not found, bad data file, etc. — handle/log and continue
    print(f"Spending Agent failed: {e}")
```

`analyze_customer()` returns:
```json
{
  "customer_id": "CUST59072A",
  "avg_monthly_income": 27700.0,
  "avg_monthly_spending": 21956.4,
  "monthly_surplus": 5743.6,
  "months_analyzed": 6,
  "overspending_categories": [],
  "analyzed_by": "Spending Agent"
}
```

For the human-readable coach message on top of this analysis, call:
```python
from spending_agent_local import generate_coach_message

result = generate_coach_message(customer_id)
# {"customer_id": ..., "coach_message": "...", "guardrail_warning": None or str}
```

## Functions reference (spending_agent_tools.py)
- `get_customer_transactions(customer_id, months=None)` → DataFrame of a customer's transactions.
- `calculate_monthly_surplus(customer_id)` → dict with avg income/spending/surplus.
- `detect_overspending_categories(customer_id, threshold_pct=30.0, scope="any_month")` → list of flagged month/category spikes vs. the customer's own baseline.
  - **Important:** default `scope="any_month"` checks the customer's ENTIRE history for anomalies. Using `scope="latest_month"` only checks the most recent month — a real anomaly earlier in the history will be silently missed. `analyze_customer()` and the demo both use the safe default; make sure any new caller does too.
- `categorize_transaction_description(description)` → guesses a category from raw text (useful if a real bank feed doesn't pre-label categories).
- `analyze_customer(customer_id)` → **single entry point** combining everything above, ready for the Coordinator.

All functions raise `SpendingAgentError` (not generic exceptions) on bad
input or missing data, so the Coordinator can catch one specific error type.

## Guardrails (spending_agent_local.py)
The local LLM (Llama 3.1 8B) is noticeably weaker than a hosted model,
and testing surfaced 3 distinct real failure modes it can produce. Each
has its own check, all run automatically inside `generate_coach_message()`:

| Check | Catches | Found via |
|---|---|---|
| `check_for_hallucinated_categories` | Model invents a spending category/amount that was never actually flagged | Real bug found in testing |
| `check_for_ignored_real_flags` | Model reassures the customer everything is normal while silently dropping a real, data-backed overspending flag | Real bug found in testing (any wording, not just specific phrases) |
| `check_for_surplus_deficit_contradiction` | Model reuses a real number but mislabels it (e.g. calls average spending a "deficit" when the customer has a positive surplus) | Real bug found in testing |

If any check fires, `generate_coach_message()` returns a non-`None`
`guardrail_warning` string describing exactly what looked wrong — this
should be surfaced to a human/logged, not silently dropped, in any
production use.

## Known limitations / things NOT done yet
- Not yet wired into the LangGraph Coordinator (this agent works standalone).
- English only for now (Arabic support was intentionally postponed).
- `detect_overspending_categories` compares each month against the
  customer's own other months — it needs at least 2 months of history
  to say anything.
- Guardrails are keyword/rule-based, not exhaustive — they catch the 3
  specific failure patterns found during testing, not every possible
  hallucination. Treat them as a safety net, not a guarantee.