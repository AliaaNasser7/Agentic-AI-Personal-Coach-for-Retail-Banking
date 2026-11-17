# Recommendation Agent → Coordinator Handoff

**From:** [Your name] , Recommendation Agent
**To:** Ahella , Architecture & Orchestrator Lead

---

## 1) The function you'll call

```python
from recommendation_agent import run_recommendation_agent

recommendation = run_recommendation_agent(spending_output, goals_output, simulation_output)
```

Same name and same three positional args as the stub currently in `agent_stubs.py`, so **nothing else in `orchestrator.py` needs to change**. Replace the stub's import with the line above and delete the old stub function.

**You don't need to call anything else.** This single function takes the three agents' outputs and returns the final recommendation, ready to drop into the shared State.

If you want to pass a `customer_id` explicitly (standalone / test use), there is also:

```python
from recommendation_agent import run
recommendation = run(customer_id, spending, goals, simulation)
```

## 2) What goes in

Three plain dicts, exactly as the other agents already return them (full field-by-field examples in `contracts.py`):

| Parameter | Comes from | Can be `None`? |
|---|---|---|
| `spending_output` | `spending_agent_tools.analyze_customer()` | Yes |
| `goals_output` | `GoalAgent.recommend()` | Yes |
| `simulation_output` | `build_simulation_output()` | Yes |

If an upstream agent failed, pass `None` for it. The agent degrades gracefully instead of crashing, and reports what it actually used in `used_inputs`.

## 3) Output shape (JSON-ready, verified)

```json
{
  "customer_id": "CUSTE2483D",
  "summary": "You currently have a monthly surplus of 5,142.08. Your top goal is \"Emergency Fund\". Overall, you're saving 11,653.91 less per month than your goals need.",
  "recommendations": [
    {
      "title": "Accelerate Your \"Emergency Fund\" Goal",
      "reason": "Your \"Emergency Fund\" goal needs about 12,200 per month to reach 73,200 in 6 months. ...",
      "estimated_impact": "Frees up about 1,127.34 per month (9.20% of what this goal requires).",
      "action": "Convert \"CINEMA TICKET PURCHASE\" (1,250.28, Entertainment) into a 12-month installment plan, ..."
    }
  ],
  "goal_impact": {
    "goal_name": "Emergency Fund", "target_amount": 73200.0, "duration_months": 6,
    "monthly_required": 12200.0, "current_monthly_pace": 546.09,
    "projected_months_to_reach": 134.0, "on_track": false
  },
  "priority": {"score": 38.5, "label": "Medium", "model": "RandomForestRegressor"},
  "used_inputs": {"spending": true, "goals": true, "simulation": true},
  "analyzed_by": "Recommendation Agent"
}
```

- If there is not enough information, `recommendations` is `[]` (not null) and `goal_impact` / `priority` are `null`.
- `priority` is `null` if the ML model or scikit-learn isn't available. Everything else still works.

## 4) No LLM, no Ollama

This agent is **rule-based**: every number in the output is read directly from the three inputs, so there is no hallucination risk and no guardrail layer is needed. Nothing needs to be running locally for it to work.

The `priority` field is a small, **additive** ML score (RandomForest, 0-100, High / Medium / Low). It never changes which recommendations are produced or their order.

## 5) Errors

The agent does not raise on missing or partial input. Pass `None` for any failed agent and check `used_inputs` in the result to see what was used. If all three are `None`, you get the "not enough information" result described above.

## 6) Things we need to align on

-  The agent finds `customer_id` on whichever input dict has it. Will `customer_id` always reach it in the dataset format (`CUSTXXXXXX`)?
-  If `priority` is `High`, should the Coordinator / dashboard do anything special (e.g. surface it first), or just display it?
-  Primary goal: when a customer has several goals, the agent leads with the `needs_action` goal that has the largest `monthly_gap` (otherwise the first goal). Is that the rule the team wants?

## 7) Reference files

- `README.md` — full documentation of inputs, outputs, and the ML layer
- `contracts.py` — exact input/output shapes
- `demo_recommendation_agent.py` — a runnable walkthrough showing 3 scenarios (standalone, no other agents needed)
- `local_demo.py` — live end-to-end run against the REAL Spending / Goals / Simulation agents
- `test_recommendation_agent.py` — 15 unit tests (`python -m unittest recommendation_agent.test_recommendation_agent -v`, run from the project root)
