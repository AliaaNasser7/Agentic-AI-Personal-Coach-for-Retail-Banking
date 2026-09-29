# Agentic AI Personal Coach — full pipeline, now with the real Goals & Simulation agents

This folder is a standalone, runnable version of the graduation project.
It previously shipped with placeholder Goals and Simulation agents (no
branch existed for Goals; the `Simulation_agent` branch was empty). Both
have now been replaced with the actual implementations your teammates
built, pulled from:

**https://github.com/AliaaNasser7/Agentic-AI-Personal-Coach-for-Retail-Banking**

| Branch | What's on it | What I did |
|---|---|---|
| `goal_agent` | Real "Instant Installment Advisor" — computes disposable income, a monthly gap per goal, and priced purchase-to-installment suggestions from the products catalog | Vendored unmodified at `agents/vendor/goal_agent.py`; wired in via `agents/goals_agent.py` |
| `Simulation_agent` | Real, tested, documented (`docs/README.md`) — balance projection, per-goal + overall feasibility, ranked spending-cut scenarios, ranked savings-product options, fact-grounded narration | Vendored unmodified at `agents/vendor/simulation_agent/` + `agents/vendor/shared/`; wired in via `agents/simulation_agent.py` |
| `spending-agent` | Real, tested (25 unit tests) — this project's `agents/spending_agent.py` was already this code, just pointed at a shared data path | Also re-vendored at `agents/vendor/spending_agent_tools.py` for full fidelity to "use the real branch code", via the same adapter pattern as the other two |
| *(no recommendation branch exists in the repo)* | — | `agents/recommendation_agent.py` keeps its original logic and flow (deterministic fan-in, no ML/LLM, graceful degradation) but now reads the real Goals/Simulation output shapes, and prefers the Goals Agent's own priced installment suggestion as the recommended action |

Nothing was pushed back to the team's GitHub repo — this is handed over
separately so you can review it before merging anything back.

## Project structure

```
project/
  data_access.py              # shared data loading — now also load_products()
  data/                       # the project's data (untouched — same generator/seed as every branch)
    customers.csv
    transactions.csv
    goals.csv
    products_catalog.json
  agents/
    vendor/                   # teammates' real code, byte-for-byte unmodified
      goal_agent.py                    # from the `goal_agent` branch
      spending_agent_tools.py          # from the `spending-agent` branch
      shared/finance_utils.py          # from the `Simulation_agent` branch
      simulation_agent/                # from the `Simulation_agent` branch
        forecasting.py, scenarios.py, products.py, narration.py, report.py, errors.py
    spending_agent.py         # adapter: points vendor/spending_agent_tools.py at data_access.py
    spending_agent_llm.py     # optional Ollama coach-message layer (unchanged)
    goals_agent.py            # adapter: wraps vendor/goal_agent.py's GoalAgent
    simulation_agent.py       # adapter: wraps vendor/simulation_agent's build_simulation_output()
    recommendation_agent.py   # fan-in agent — same logic/flow, updated to the real shapes above, plus the ML layer below
    ml/                       # ML priority-scoring layer (additive, see §7)
      features.py                 # engineers a numeric feature vector from Spending/Goals/Simulation output
      train_priority_model.py     # trains + saves the model (already run — model is checked in below)
      priority_model.py           # loads the model, scores a feature vector, fails soft if unavailable
      model/priority_model.joblib # the trained scikit-learn RandomForestRegressor
  coordinator/
    contracts.py              # reference docs of what each agent now actually returns
    orchestrator.py           # unchanged: parallel fan-out, fan-in, question router
  api.py                      # minimal Flask API
  frontend/
    index.html                # dashboard UI — "why am I seeing this" panel updated for the new shapes
  tests/
    test_recommendation_agent.py   # 10 tests, fixtures updated to the real agent shapes
```

## How to run it

You need Python 3.9+ and pip.

```bash
cd project
pip install -r requirements.txt --break-system-packages   # drop --break-system-packages if not needed on your machine
python api.py
```

Then open **http://127.0.0.1:5000** in a browser. Pick a customer from the
dropdown in the sidebar; the dashboard loads automatically.

No database and no Ollama are required for the demo to work. Ollama is
optional in two places:
- **Spending tab coach message** (`spending_agent_llm.py`) — if Ollama is
  running with `llama3.1:8b` pulled, it's used; otherwise the orchestrator
  falls back to a plain sentence built from the real numbers.
- **Goals Agent per-goal `customer_message`** — off by default (matches
  the vendored `GoalAgentConfig`'s own default, and avoids a
  connection-refused log line per goal when Ollama isn't running). Set
  `GOALS_AGENT_USE_LLM=1` before starting `api.py` to turn it on; it
  falls back to `null` cleanly if Ollama isn't reachable either way.
- **Simulation Agent's `insight_text`** narration always attempts an
  Ollama polish pass and falls back to its own fact-grounded sentences
  silently if Ollama isn't running — no extra flag needed.

### Running just the Recommendation Agent's tests

```bash
cd project
python -m unittest tests.test_recommendation_agent -v
```

All 10 tests should pass.

### Running any agent standalone (no server needed)

```bash
cd project
python agents/spending_agent.py
python agents/goals_agent.py
python agents/simulation_agent.py
python coordinator/orchestrator.py     # full dashboard for one customer, printed as JSON
```

## 1) What each agent now returns

See `coordinator/contracts.py` for full, real examples of every shape
below (pulled from an actual run against `data/`, not hand-written).

**Spending Agent** — `agents.spending_agent.analyze_customer(customer_id)`
— unchanged: `avg_monthly_income`, `avg_monthly_spending`,
`monthly_surplus`, `months_analyzed`, `overspending_categories[]`.
Raises `SpendingAgentError` for a bad/missing customer_id.

**Goals Agent** — `agents.goals_agent.run(customer_id)` — now wraps the
real `GoalAgent`. Each goal carries its *this-month* disposable income,
`monthly_gap`, a `status` of `"on_track"` or `"needs_action"`, and (when
there's a gap) a list of priced `suggestions` — real transactions that
could be converted into an installment plan, with the tenor, fee, and
monthly cash freed already computed. `duration_months`/`start_date` are
added by the adapter from `goals.csv`, since the real agent's own output
doesn't include them. Raises `GoalsAgentError` if the customer has no
goals or the agent failed to initialize.

**Simulation Agent** — `agents.simulation_agent.run(customer_id, spending, goals)`
— now wraps the real `build_simulation_output()`. Returns a 6-month
balance `baseline_projection`, per-goal **and** overall feasibility
(`goals.per_goal[]`, `goals.overall`, each with an `"ahead"` /
`"on track"` / `"behind"` / `"no_goals_set"` status based on actual
saving history — not just this month), ranked `top_scenarios` (spending
cuts to try), ranked `top_products` (savings/investment options, split
into `flexible` vs. `locked`), and a fact-grounded `insight_text`
paragraph. Raises `SimulationAgentError` on an unknown customer or bad
input, matching the `SpendingAgentError`/`GoalsAgentError` pattern.

## 2) What the Recommendation Agent uses — and what changed

It still takes exactly these three dicts — nothing else, no re-reading
of CSVs — and never recomputes surplus, overspending, or goal math
itself, because that would duplicate work the other agents already did:
`spending` (or `None`), `goals` (or `None`), `simulation` (or `None`).

What's new, now that Goals and Simulation carry richer information:
- The goal recommendation's **action** now prefers the Goals Agent's own
  priced installment suggestion ("convert this specific purchase into a
  12-month plan") over a generic "reduce this category" instruction,
  because it's a concrete, ready-to-execute option rather than an
  instruction the customer has to figure out themselves.
- The goal recommendation's **reason** cites both signals when both are
  present: the Goals Agent's this-month disposable-income gap, *and* the
  Simulation Agent's longer-horizon feasibility based on actual saving
  history — surfaced together rather than one silently overriding the
  other (this is exactly the "conflicting information" test case).
- A new **savings-product recommendation**, sourced from Simulation's
  ranked product options, is now available as a fallback for a customer
  with a healthy surplus but no goal gap to close — the old version had
  nothing more specific to offer in that case than a generic "keep
  saving" line.
- The **goal_impact** block's `current_monthly_pace` and
  `projected_months_to_reach` now come from the Simulation Agent's
  per-goal feasibility (drawn from real saving history) rather than a
  single surplus figure.

Degradation rules are unchanged: any of the three inputs can be `None`,
and the agent falls back step by step (goal+spending combined → goal
only → spending only → savings-product → general advice → "not enough
information") rather than crashing or inventing a concern that isn't in
the data.

## 3) Where it's connected

`coordinator/orchestrator.py` (unchanged — it only ever passed whatever
dict each agent returned along to the next one, so it didn't need to
change for this integration):

```
Spending Agent  ─┐
                  ├─▶ Simulation Agent ─▶ Recommendation Agent
Goals Agent     ─┘
```

Spending and Goals run in parallel (`ThreadPoolExecutor`, 2 workers)
since neither depends on the other. Simulation now actually recomputes
cash flow and goal feasibility itself, directly from `transactions.csv`/
`goals.csv` (see `agents/vendor/shared/finance_utils.py`'s note on
avoiding cross-agent rounding drift), rather than trusting numbers a
sibling agent already rounded — but it still runs after Spending/Goals
in the pipeline, matching the original dependency shape. Recommendation
is the final fan-in step, exactly as in the brief's diagram.

There's also a lightweight, keyword-based question router
(`orchestrator.handle_question`) for the chat-style demo: a plain lookup
question ("How much did I spend on restaurants?") only calls the
Spending Agent and skips Recommendation entirely; a broader question
("How can I save more money?") calls all three agents and then
Recommendation. This is simple keyword matching on purpose, not NLP.

## 4) How the vendored code is wired in (if you're extending this)

Each of the three vendored files/packages was written as if it were the
only thing in the project (its own `output/` folder, its own top-level
`shared`/`simulation_agent` package names). Rather than editing the
teammates' logic to fit this project, each adapter (`agents/spending_agent.py`,
`agents/goals_agent.py`, `agents/simulation_agent.py`) does the minimum
needed to reuse it as-is:
- adds `agents/vendor/` to `sys.path` so the vendored modules import the
  way they were written to (`import shared`, `import simulation_agent`,
  plain `import spending_agent_tools` / `goal_agent`) — these resolve to
  different `sys.modules` entries than this project's own
  `agents.spending_agent` etc., so there's no naming collision;
- for Spending, monkey-patches the vendored module's two path constants
  (`TRANSACTIONS_PATH`, `CUSTOMERS_PATH`) to point at `data_access.py`'s
  paths instead of a local `output/` folder — the same approach the
  team's own `coordinator-agent` branch used in `agent_stubs.py`;
  Goals/Simulation take the paths as constructor/function arguments, so
  no monkey-patching is needed there;
- translates each vendored agent's own error convention into the
  `*AgentError` exception type the Coordinator already knows how to
  catch (`SimulationAgentError` wraps the vendored `errors.SimulationAgentError`
  and its `{"error": ...}` dict convention; `GoalsAgentError` wraps
  `GoalAgent.recommend()`'s `{"error": ...}` convention).

If a teammate pushes an update to their branch, the fix is almost always
just re-copying their file(s) into `agents/vendor/` — the adapters
shouldn't need to change unless the public function/class names changed.

## 5) Demo walkthrough

**"How can I save more money?"** (customer Allison Hill, `CUSTE2483D`)
→ Agents used: Spending, Goals, Simulation, Recommendation
→ Goals Agent finds her Emergency Fund goal is short 3,675.40 this month
  after essential expenses, and finds a cinema-ticket purchase that could
  become a 12-month installment, freeing up 1,127.34/month.
→ Simulation Agent finds that, based on her actual 6-month saving
  history (not just this month), she's saving only 546/month toward a
  goal that needs 12,200/month — 134 months to reach it at this pace
  instead of the planned 6.
→ Recommendation Agent combines these: *"Convert 'CINEMA TICKET
  PURCHASE' (1,250.28, Entertainment) into a 12-month installment plan,
  and redirect the freed-up cash toward your 'Emergency Fund' goal"* —
  with both the this-month gap and the longer-horizon shortfall cited in
  the reason, and the installment's cash-freed as the stated impact.
→ UI shows this as the Top Recommendation card, with the Goal Impact bar
  reflecting her real saving-history pace against what the goal requires.

**"How much did I spend on restaurants?"** (any customer)
→ Agents used: Spending Agent only — a direct lookup, Recommendation is
  skipped entirely.

**"How can I reach my goal faster?" / "Can I afford this goal?"**
→ Same full pipeline as the first example.

## 6) UI states

The dashboard (`frontend/index.html`) implements the same four states as
before, toggled by JS based on the API response: `app-loading` (spinner
while fetching), `app-content` (recommendation available),
`app-insufficient` (Recommendation Agent returned zero recommendations),
`app-error` (API call failed or customer not found). The "Why am I
seeing this?" panel was updated to read the real Goals/Simulation
shapes — it now surfaces the Goals Agent's this-month gap and
installment-option count, and the Simulation Agent's full `insight_text`
narrative instead of a single projected-months number.

## 7) ML layer

`agents/ml/` adds a trained **scikit-learn `RandomForestRegressor`**
that scores every recommendation result with a `priority` field:

```json
"priority": {"score": 50.3, "label": "Medium", "model": "RandomForestRegressor"}
```

**What it predicts:** a 0–100 "how urgently does this situation call for
action" score (High ≥60, Medium 30–59, Low <30), shown as a badge next
to the summary and used in the UI's tooltip.

**What it's trained on:** `agents/ml/train_priority_model.py` runs the
real Spending → Goals → Simulation pipeline for every customer in
`data/customers.csv`, extracts a 10-feature vector per customer
(`agents/ml/features.py` — surplus ratio, overspend severity, goal-gap
ratio, months-behind ratio, best available spending-cut gain, etc.),
and trains against a transparent, documented composite-urgency formula
(see that file's docstring) since there's no historical
acceptance/outcome data in this project to learn from instead. 5-fold
cross-validation gets R² ≈ 0.80. The trained model is checked in at
`agents/ml/model/priority_model.joblib`, so nothing needs to be
retrained to run the demo — re-run `python -m agents.ml.train_priority_model`
only if you change the feature set or the label formula.

**What it does *not* do:** decide which recommendations are shown, their
order, or any of their wording — that's all still the same deterministic,
rule-based logic described in §2, grounded entirely in real numbers from
Spending/Goals/Simulation. The ML layer is a second, additive lens
("how urgent, overall?") sitting alongside the first ("what specifically
should they do?"), not a replacement for it — this keeps the
recommendation *content* itself free of any model-hallucination risk,
while still giving the project a genuine, trained, cross-validated ML
component. `tests/test_recommendation_agent.py`'s `TestMLPriorityLayer`
class checks exactly this: swapping the ML layer on/off must never
change `recommendations` or `goal_impact`, only `priority`.

If the model file is missing or scikit-learn isn't installed,
`priority` is simply `None` and everything else keeps working — see
`agents/ml/priority_model.py`'s fail-soft loader.

## 8) Known limitations (be upfront about these if asked)

- The "primary goal" picked when a customer has more than one goal is
  still simply the one with the soonest deadline — a clear, explainable
  rule, not a scoring model.
- The Goals Agent's gap/suggestions are computed from a single month's
  transactions (the most recent, by default); the Simulation Agent's
  feasibility is computed from the customer's full saving history. The
  Recommendation Agent surfaces both rather than picking one as "more
  correct" — they're answering slightly different questions ("can I
  close this month's gap with one purchase?" vs. "am I on pace overall?")
  and can legitimately disagree, which is itself useful information for
  the customer.
- The coach-message LLM layer (`spending_agent_llm.py`) is a trimmed
  port of the original — the full 3-check guardrail system from the real
  `spending_agent_local.py` on the `spending-agent` branch is more
  thorough than what's here; if you want the identical guardrails, vendor
  that file in directly the same way the other three were.
- Model consistency across agents was already flagged as an open item on
  the `Simulation_agent` branch's own docs (Spending defaults to
  `llama3.1:8b`, Goals to `llama3.2`, Simulation to `llama3`) — this
  integration didn't change any of those defaults; worth a team decision
  before a live demo if the LLM layers are turned on.
- The ML priority score's training labels come from a documented
  formula (see `agents/ml/train_priority_model.py`), not real historical
  outcomes — be upfront about this if asked "what is it actually
  predicting?" It's a learned, generalizable approximation of a
  transparent urgency rule, ready to be retrained on real outcome data
  (missed goals, overdrafts, accepted vs. ignored recommendations) once
  that exists, without changing anything else in the pipeline.
