# Recommendation Agent

This folder is **only** the Recommendation Agent — the fan-in step that
takes Spending, Goals, and Simulation's real output and turns it into
one clear, explainable recommendation. It does not contain or modify
any of the other three agents' code, so merging this branch shouldn't
touch a single file that isn't in this folder.

## What's in here

```
recommendation_agent/
  __init__.py            # public API: run_recommendation_agent, run
  core.py                 # the actual logic (rule-based, no LLM, no hallucination risk)
  contracts.py             # documents the exact input/output shapes (reference only)
  ml/
    features.py                # turns spending/goals/simulation dicts into a feature vector
    priority_model.py          # loads the trained model, scores a feature vector
    train_priority_model.py    # (re)trains the model -- optional, see below
    model/priority_model.joblib  # the trained model, already included
  tests/
    test_recommendation_agent.py   # 15 tests, fully self-contained (no dependency on the other agents)
```

No data, no vendored copies of anyone else's code, no Flask app, no
frontend — just this agent and what it needs to run and be tested on
its own.

## 1. What it expects as input

Three plain dicts, matching the real output of the other three agents
exactly (see `contracts.py` for full field-by-field examples):

| Parameter | Comes from | Can be `None`? |
|---|---|---|
| `spending` | `spending_agent_tools.analyze_customer()` (spending-agent branch) | Yes |
| `goals` | `GoalAgent.recommend()` (goal_agent branch) | Yes |
| `simulation` | `build_simulation_output()` (Simulation_agent branch) | Yes |

Any of the three can be missing (that agent failed, or wasn't needed
for a given question) — the agent degrades gracefully rather than
crashing, and says in the output which inputs it actually used
(`used_inputs`).

**Note on "primary goal":** the real Goals Agent returns *every* goal
the customer has, fully analyzed, but doesn't itself pick one to lead
with when there's more than one. Picking one is a fan-in judgment call,
so it happens inside this agent (`_pick_primary_goal` in `core.py`):
whichever goal has `status == "needs_action"` with the largest
`monthly_gap`; if everything's already on track, the first goal. Simple
and explainable, not a scoring model — easy to change later if the team
wants a different rule.

## 2. What it returns

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

- **`recommendations`** is rule-based and fully traceable to real
  numbers from the three inputs — no LLM call, no invented figures.
  Prefers the Goals Agent's own priced installment suggestion as the
  action when there is one; falls back through spending-only →
  savings-product → general advice as inputs get thinner; never
  fabricates a concern that isn't in the data.
- **`priority`** is a small, *additive* ML layer (see §4) — a trained
  model's read on "how urgent is this overall," shown alongside the
  recommendations. It never changes which recommendations are produced
  or their order. If the model file or scikit-learn isn't available,
  this is simply `null` and everything else still works.

## 3. How to wire it into the Coordinator

The Coordinator's `agent_stubs.py` currently has this stub:

```python
from contracts import EXAMPLE_RECOMMENDATION_OUTPUT

def run_recommendation_agent(spending_output, goals_output, simulation_output):
    return EXAMPLE_RECOMMENDATION_OUTPUT
```

Swap it for this — same name, same three positional args, nothing else
in `orchestrator.py` needs to change:

```python
from recommendation_agent import run_recommendation_agent
```

...and delete the old stub function (the import above already *is* the
function with the exact right signature). That's the entire integration.

## 4. The ML layer (what it is, and isn't)

`priority` is a trained **scikit-learn `RandomForestRegressor`** that
scores each result 0–100 for urgency (`High` ≥60, `Medium` 30–59, `Low`
<30). It's trained on a 10-feature vector (`ml/features.py` — surplus
ratio, overspend severity, goal-gap ratio, months-behind ratio, etc.)
against a transparent, documented composite-urgency formula (see
`ml/train_priority_model.py`'s docstring) — there's no historical
acceptance/outcome data in the project yet to train on instead, and
the script says so plainly.

**It does not decide what the recommendations say or which ones show
up** — that's still 100% the rule-based logic in `core.py`. It's a
second, additive lens sitting next to the first, which is exactly what
`tests/test_recommendation_agent.py`'s `TestMLPriorityLayer` checks:
turning the ML layer off must never change `recommendations` or
`goal_impact`, only `priority`.

The trained model is already included (`ml/model/priority_model.joblib`)
— you don't need to run anything to use it. `ml/train_priority_model.py`
is only for retraining later, and only runs from inside the fully
merged repo (it needs the real Goals/Simulation/Spending code
importable, the same way `agent_stubs.py` already does it).

## 5. How to run it on your end, to make sure it works

### A) Unit tests — no other agent's code needed (do this first)

These fully exercise the agent's own logic against fixtures shaped
exactly like the real agents' output, so they prove the agent is
correct completely on their own:

```bash
cd Agentic-AI-Personal-Coach-for-Retail-Banking    # wherever recommendation_agent/ lives
pip install scikit-learn joblib --break-system-packages   # drop the flag if not needed on your machine
python -m unittest recommendation_agent.tests.test_recommendation_agent -v
```

You should see `Ran 15 tests ... OK`.

### B) Live end-to-end check against the real other three agents (optional, extra confidence)

This runs the actual Spending/Goals/Simulation code — not fixtures —
and feeds their real output through this agent, so you can see one
real, live result before merging:

1. Unzip (or `git checkout`) the three branches as sibling folders next
   to `recommendation_agent/`:
   ```
   <project root>/
     recommendation_agent/
     scripts/
     goal_agent/            <- goal_agent branch
     spending-agent/         <- spending-agent branch
     Simulation_agent/       <- Simulation_agent branch
   ```
   (A GitHub branch zip nests everything one folder deeper — move that
   inner folder's contents up one level so e.g. `goal_agent.py` sits
   directly inside `goal_agent/`.)
2. `pip install pandas requests scikit-learn joblib --break-system-packages`
3. `python scripts/local_demo.py` (or `python scripts/local_demo.py CUSTE2483D` for a specific customer)

It prints which of the three real agents succeeded, then the full,
real Recommendation Agent output as JSON. `scripts/local_demo.py` is a
dev-only convenience, not part of what needs to be merged — delete it,
keep it, or move it into the Coordinator's own scripts folder, whichever
the team prefers.
