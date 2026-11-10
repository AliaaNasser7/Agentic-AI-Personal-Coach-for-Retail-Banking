# Simulation Agent — Agentic AI Personal Coach for Retail Banking

Part of the 5-agent architecture (Coordinator, Spending, Goals, Simulation, Recommendation).
This module is the **Simulation Agent**: the only agent that looks forward instead of at
the past/present — balance forecasts, goal feasibility, ranked "what-if" scenarios, and
ranked savings-product options.

Everything runs **locally**. No cloud APIs — data generation uses Faker, narration uses a
local LLM via [Ollama](https://ollama.com).

## Folder structure

```
project/
├── data/
│   └── generate_data.py       # synthetic dataset generator (customers/transactions/goals/products)
├── agents/
│   ├── run_simulation.py      # local entry point — run this to test/demo
│   ├── shared/
│   │   └── finance_utils.py   # cash-flow + savings-rate math — import this from OTHER agents too
│   │                          # (e.g. the Goals Agent), don't recompute it locally
│   └── simulation_agent/      # the Simulation Agent, split by concern
│       ├── __init__.py        # exposes build_simulation_output, run_full_simulation_report
│       ├── forecasting.py     # step 2 + 3: balance projection, goal feasibility
│       ├── scenarios.py       # step 4 + 4b: what-if simulation + ranking
│       ├── products.py        # step 5 + 5b: product simulation + ranking
│       ├── narration.py       # step 6: fact-based summary, optional local-LLM polish
│       ├── errors.py          # SimulationAgentError — raised by forecasting/scenarios/products
│       │                      # on an unknown customer_id, caught by report.py
│       └── report.py          # orchestrator — ties every module together
├── ui/
│   └── simulation_tab_mockup.html   # static visual reference for the Simulation dashboard tab
├── docs/
│   └── README.md              # this file
└── output/                    # generated at runtime — CSVs, catalog JSON, sample output
    ├── customers.csv
    ├── transactions.csv
    ├── goals.csv
    ├── products_catalog.json
    └── simulation_output_sample.json
```

**Why `shared/` sits inside `agents/`, not at the project root:** it needs to be importable with zero `PYTHONPATH` setup — running `python run_simulation.py` from inside `agents/` puts `agents/` on the import path automatically, so both `shared` and `simulation_agent` resolve as plain top-level packages. If the Goals Agent also lives under `agents/goals_agent/`, it gets the same `from shared.finance_utils import ...` for free.

## Setup

```bash
conda create -n banking-coach python=3.11
conda activate banking-coach
conda install pandas numpy
pip install faker requests
```

For AI narration (optional but required to satisfy the project's "uses AI" requirement):

```bash
# install Ollama from ollama.com, then:
ollama pull llama3
```

Model used: `llama3` by default, overridable via the `SIMULATION_LLM_MODEL` environment
variable (`export SIMULATION_LLM_MODEL=llama3.2`, for example). **Worth syncing with the
team** — Spending Agent currently uses `llama3.1:8b` and Goals Agent uses `llama3.2`; three
different models across the demo is worth a deliberate team decision, not an accident.

## Running it

```bash
cd data && python generate_data.py        # creates ../output/*.csv and products_catalog.json
cd ../agents && python run_simulation.py  # runs the full pipeline, prints a report,
                                           # writes output/simulation_output_sample.json
```

To use this agent from another agent or the Coordinator, import the package rather than
running the script:

```python
from simulation_agent import build_simulation_output
from shared.finance_utils import load_data

customers, transactions, goals, products = load_data("output")
output = build_simulation_output("CUST123", transactions, goals, products)
```

## What the Simulation Agent computes

| Step | Function | Lives in | What it answers |
|---|---|---|---|
| 1 | `monthly_cash_flow` | `shared/finance_utils.py` | Avg income / spend / net per month |
| 2 | `project_balance` | `simulation_agent/forecasting.py` | Linear balance forecast N months ahead |
| 3 | `savings_rate` | `shared/finance_utils.py` | Avg amount saved per month |
| 3 | `goal_feasibility` | `simulation_agent/forecasting.py` | Is each goal (and goals combined) on pace? |
| 4 | `rank_scenarios` | `simulation_agent/scenarios.py` | Top N spending-cut scenarios, ranked by monthly gain |
| 5 | `rank_product_options` | `simulation_agent/products.py` | Top N savings-product options, split into flexible (no lock-in) vs locked (fixed-term) |
| 6 | `narrate_report` | `simulation_agent/narration.py` | Turns the numbers into a short paragraph — numbers are always built in Python first, the LLM is only allowed to rephrase them, never compute or invent them |
| 7 | `build_simulation_output` | `simulation_agent/report.py` | Assembles everything above into the single JSON contract below |

`monthly_cash_flow` and `savings_rate` live in `shared/`, not inside the `simulation_agent`
package, specifically so the Goals Agent can import the exact same functions instead of
reimplementing its own version of "how much does this customer save per month."

## Design principles worth knowing before extending this

- **Math is deterministic, narration is not.** All financial figures come from plain pandas/Python. The LLM (`narrate_report`) is only ever handed already-correct sentences and told not to change any number in them — this was a deliberate fix after an earlier test run showed a local LLM hallucinating a month count, a percentage, and a time horizon when given raw numbers to describe freely. See `build_fact_sentences` for the guaranteed-correct fallback.
- **Per-goal status is proportional, not independent.** `avg_saved` is one pool of money shared across all the customer's goals — `goal_feasibility` allocates it proportionally to each goal's own required amount, so per-goal badges never contradict the `overall` status (an earlier version checked each goal against the *full* savings amount independently, which could show every goal "ahead" while the combined total was actually short).
- **Product ranking is split by liquidity**, not one merged list — a high-rate 5-year certificate should never silently outrank a same-day-access savings account just because it has a bigger raw interest number over the same simulated horizon.

## Known edge cases (tested)

| Input | Behavior |
|---|---|
| Unknown `customer_id` (no transaction history) | `project_balance`, `simulate_scenario`, and `rank_product_options` each raise `simulation_agent.errors.SimulationAgentError` directly (catchable by type, matching Alia's `SpendingAgentError` pattern). `build_simulation_output` catches it and returns `{"error": "..."}` instead — check for the `"error"` key before reading other fields. `run_full_simulation_report` catches it and prints, returning `None` |
| Valid customer, zero goals in `goals.csv` | `goals.overall.status` is `"no_goals_set"` (not `"on track"` — that would be a false claim). `insight_text` pivots to highlighting their monthly surplus and the top savings-product options instead |
| Valid customer, one goal only | Works normally — proportional allocation with one goal just allocates 100% of `avg_saved` to it |
| Customer with zero credit transactions (no Salary/income rows on file) | `avg_monthly_income` returns `0.0`, not `NaN` — `NaN` isn't valid JSON per spec and a strict parser on the receiving end would fail even though Python's own `json` module writes it silently |
| Any output, generally | All numpy scalar types (`int64`, `float64`) are converted to native Python types before `build_simulation_output` returns (`_to_native` in `report.py`) — caught by a customer whose `balance_after` column happened to load as all-integer values, which crashed a plain `json.dumps()` with `TypeError: Object of type int64 is not JSON serializable`. The `default=str` fallback in `run_simulation.py` was silently masking this by turning numeric fields into strings instead of numbers — worth checking any other `json.dump(..., default=str)` calls in the project for the same silent-masking risk |
| Customer whose spending has zero overlap with the discretionary categories tried (e.g. no Dining/Entertainment/Shopping/Subscriptions at all) | `rank_scenarios` now filters out non-improving results — previously it could return a fake "best option" showing `+0` monthly gain. `insight_text` says "No spending category currently offers a meaningful cost-cutting opportunity" instead |
| Customer whose balance/surplus doesn't meet any product's `min_amount` | `top_products` is `{"flexible": [], "locked": []}` — `insight_text` now says so explicitly instead of silently omitting that section |
| Customer spending more than they earn (negative `avg_monthly_net`) | Projections correctly go negative and keep working. The "no goals yet, good position to start one" message now branches on sign — a negative-net customer gets "you're spending more than you bring in... worth addressing that before setting a savings target" instead of being told they're in a good position |
| `months_ahead <= 0` | Previously crashed with a raw `IndexError` inside narration. Now validated in `_compute()` and raises `SimulationAgentError` like the other bad-input cases |
| `top_n = 0` | Already handled correctly — empty scenario/product lists, no crash |

## JSON output contract

`build_simulation_output(customer_id, transactions_df, goals_df, products, months_ahead=6, top_n=3)`
is the function other agents / the Coordinator should call — it has no side effects (no prints)
and returns a plain dict that's safe to `json.dumps()` or return from an API endpoint.

```jsonc
{
  "agent": "simulation",
  "schema_version": "1.0",
  "customer_id": "CUSTTEST1",
  "months_ahead": 6,
  "cash_flow": {
    "avg_monthly_income": 15000.0,
    "avg_monthly_spend": 11400.0,
    "avg_monthly_net": 3600.0
  },
  "baseline_projection": {
    "last_known_balance": 26600.0,
    "last_known_date": "2026-06-15",
    "avg_monthly_net": 3600.0,
    "projection": [
      { "date": "2026-07-15", "projected_balance": 30200.0 }
      // ... one entry per month, up to months_ahead
    ]
  },
  "goals": {
    "per_goal": [
      {
        "goal_name": "Vacation",
        "target_amount": 9000,
        "planned_months": 9,
        "required_monthly": 1000.0,
        "allocated_monthly_savings": 900.0,
        "actual_months_needed": 10.0,
        "status": "on track"   // "ahead" | "on track" | "behind"
      }
    ],
    "overall": {
      "avg_monthly_saved": 1500.0,
      "total_required_monthly": 1500.0,
      "status": "on track",   // "on track" | "behind"
      "monthly_shortfall": 0.0
    }
  },
  "top_scenarios": [
    {
      "scenario": { "Dining/Restaurants": -0.2 },
      "adjusted_avg_monthly_net": 3780.0,
      "projection": [ /* same shape as baseline_projection.projection */ ],
      "monthly_gain": 180.0
    }
    // up to top_n, sorted best-first by monthly_gain
  ],
  "top_products": {
    "flexible": [
      {
        "product": "Regular Savings Account",
        "annual_rate": 0.09,
        "contribution_type": "lump_sum",   // "lump_sum" | "monthly"
        "contribution_amount": 26600.0,
        "months_simulated": 6,
        "total_contributed": 26600.0,
        "final_balance": 27819.67,
        "interest_earned": 1219.67,
        "warning": null
      }
      // up to top_n
    ],
    "locked": [
      {
        "product": "5-Year Fixed Deposit Certificate",
        "annual_rate": 0.185,
        "contribution_type": "lump_sum",
        "contribution_amount": 26600.0,
        "months_simulated": 6,
        "total_contributed": 26600.0,
        "final_balance": 29157.3,
        "interest_earned": 2557.3,
        "warning": "Note: 5-Year Fixed Deposit Certificate locks funds for 60 months. You simulated only 6 months - early withdrawal isn't modeled here."
      }
      // up to top_n
    ]
  },
  "insight_text": "You're earning about 15,000 per month and spending about 11,400, leaving a net of 3,600 per month. ..."
}
```

A real generated example lives at `output/simulation_output_sample.json` after running the script.

## Open items for integration with the rest of the team

- ~~Confirm the Coordinator calls `build_simulation_output`, not `run_full_simulation_report`~~ — **confirmed by the team.**
- Confirm `schema_version` handling if this contract changes later — bump the string so the Recommendation Agent can detect a mismatch.
- **Shared cash-flow math**: `monthly_cash_flow`/`savings_rate` now live in `agents/shared/finance_utils.py` and round to 2 decimals at the source (this fixed an observed rounding drift — `22557.92` vs `22557.923333...` for the same customer, computed independently by two agents). Aya's Spending Agent and Goals Agent should import from here rather than keep their own local versions, or the drift comes back.
- **Model consistency**: this agent defaults to `llama3` (overridable via `SIMULATION_LLM_MODEL`). Spending uses `llama3.1:8b`, Goals uses `llama3.2`. Worth a team decision on whether to standardize before the demo, or explicitly note per-agent model choice is intentional.