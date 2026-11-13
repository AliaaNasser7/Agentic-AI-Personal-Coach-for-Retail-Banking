# Coordinator Agent — Ahella Basem

## Role
The Coordinator is the single entry point to the system. For one customer ID it
calls the four specialist agents (Spending, Goals, Simulation, Recommendation),
combines their outputs, and returns them as one JSON response through a FastAPI
endpoint. It does no financial analysis of its own. Its jobs are orchestration,
failure handling, and keeping the other agents' outputs consistent.

## Status

| Agent | Owner | Status | Called through |
|---|---|---|---|
| Spending | Alia | Real, wired in | `analyze_customer()` + `generate_coach_message()` |
| Goals | Aya | Real, wired in | `GoalAgent.recommend()` |
| Simulation | Ahlam | Real, wired in | `build_simulation_output()` |
| Recommendation | Alaa | Real, wired in | `run_recommendation_agent()` |

chat_agent.py holds an experimental chatbot (POST /chat/{customer_id}). It routes questions to the agents by keyword and makes one Ollama call (llama3.1:8b). It is not final and is not covered by the merge checklist.

## Files in this folder

| File | Purpose |
|---|---|
| `main.py` | FastAPI app. Routes: `GET /` (health check) and `GET /dashboard/{customer_id}` |
| `orchestrator.py` | `load_dashboard(customer_id)` calls the agents in order and combines the results |
| `agent_stubs.py` | Imports each real agent and wraps it so failures are handled in one place. The filename is historical, since these are no longer stubs |
| `data_access.py` | Loads `customers.csv` and looks up one customer |
| `contracts.py` | Reference only: example output shapes of each agent. Nothing imports it |
|`chat_agent.py`|

## API

`GET /dashboard/{customer_id}`

```
{
  "customer": { ...row from customers.csv... },
  "tabs": {
    "spending":   { ... } or { "error": "...", "analyzed_by": "..." },
    "goals":      { ... } or { "error": "...", "analyzed_by": "..." },
    "simulation": { ... } or { "error": "...", "analyzed_by": "..." },
    "overview":   { ... } or { "error": "...", "analyzed_by": "..." }
  }
}
```

- Unknown customer ID: HTTP 404 with a `detail` message.
- If one agent fails, only that tab contains the error dict. The other tabs still return.
- Call order is sequential: Spending, Goals, Simulation, Recommendation. Simulation does not
  use the Spending or Goals outputs (the parameters exist only to keep the call signature).
- Before calling Recommendation, the Coordinator converts any failed agent's `{"error": ...}`
  to `None`. The Recommendation Agent expects `None` for a failed agent and reports it in
  `used_inputs`.

### Failure behavior (the dashboard still loads)
- **Spending:** if the LLM call fails or times out, or a guardrail fires, the Coordinator
  replaces `coach_message` with a plain sentence built from the numbers
  (`fallback_spending_message`) and prints a `[GUARDRAIL]` line to the terminal.
- **Goals:** if Ollama or the model is unavailable, `customer_message` is `null`.
- **Simulation:** if the LLM is unavailable, `insight_text` is built from the fact sentences only.
- **Recommendation:** rule-based, no LLM call.

## Environment setup (Windows, Python 3.12)

```powershell
cd coordinator_agent
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
venv\Scripts\Activate.ps1
pip install pandas requests fastapi uvicorn scikit-learn joblib --timeout 120 --retries 5
```

### Required: make the sibling agent folders importable (`.pth` file)
Run once from `coordinator_agent`, after creating the venv:

```powershell
$root = (Resolve-Path "..").Path
@($root, "$root\goals_agent", "$root\spending_agent") | Set-Content -Encoding ascii "venv\Lib\site-packages\nbe_project.pth"
```

This makes `shared`, `simulation_agent`, `recommendation_agent`, `goal_agent`, and
`spending_agent_tools`/`spending_agent_local` importable as top-level modules. The `.pth`
file lives inside the venv, so it is not committed. Every teammate recreates it after cloning.

We use this instead of `sys.path.append(...)` in code because VS Code's "organize imports"
silently moved the imports above the `sys.path` lines, which broke the file repeatedly.

### Recommended: disable import auto-organizing in VS Code
`.vscode/settings.json`:
```json
{
  "editor.codeActionsOnSave": { "source.organizeImports": "never" },
  "python.analysis.autoImportCompletions": false
}
```

### Ollama models
Each agent currently uses a different local model. All three must be pulled:

```powershell
ollama pull llama3.1:8b   # Spending
ollama pull llama3.2      # Goals
ollama pull llama3        # Simulation
```

A missing model does not crash anything. It only turns off that agent's LLM text
(Goals returns a `404` message in the terminal and `customer_message: null`).

### GPU note
If Ollama fails with `CUDA error: the provided PTX was compiled with an unsupported
toolchain`, this is an Ollama and NVIDIA-driver mismatch, not a code bug. Ollama needs
driver 570+ for GPUs with compute capability 5.0 to 6.2 (this includes entry-level MX-series
laptop GPUs). Check your driver with `nvidia-smi`, then do a clean install of a newer driver
and restart. As a fallback, `setx OLLAMA_NUM_GPU 0`, then fully quit and restart the Ollama
background app. Setting it with `$env:` in a terminal does not affect an already-running service.

## Running

```powershell
cd coordinator_agent
venv\Scripts\Activate.ps1
uvicorn main:app --reload
```

- Dashboard: `http://127.0.0.1:8000/dashboard/CUSTE2483D`
- Interactive docs: `http://127.0.0.1:8000/docs`
- At startup you should see
  `[DEBUG] Spending Agent will read customers from: ...\output\customers.csv`.
  If that path is not the shared root `output/` folder, the path override is not working.
- **One request takes 1-2 minutes** on a laptop. Three LLM-backed agents run in sequence
  and three different Ollama models are loaded and unloaded between them.

Recommendation Agent tests (from the project root):
```powershell
python -m unittest recommendation_agent.tests.test_recommendation_agent -v
```
Expected: `Ran 15 tests ... OK`.

## Folder structure for merging into main

Every agent lives in its own top-level folder, and there is exactly one shared `output/`.

```
Agentic-AI-Personal-Coach-for-Retail-Banking/
├── README.md                       (project-level, already on main; do not overwrite)
├── .gitignore                      (must include: venv/  __pycache__/  *.pyc  .env)
├── requirements.txt                (one combined file, see below)
├── data.py                         (the single data generator, already on main)
├── run_simulation.py               (Ahlam's local runner; stays at the root)
├── output/                         (the ONLY copy of the data)
│   ├── customers.csv
│   ├── transactions.csv
│   ├── goals.csv
│   └── products_catalog.json
├── shared/
│   ├── __init__.py                 (empty file)
│   └── finance_utils.py
├── spending_agent/
│   ├── spending_agent_tools.py
│   ├── spending_agent_local.py
│   └── README.md                   (plus her demo and tests, inside this folder)
├── goals_agent/
│   ├── goal_agent.py               (folder is plural, file is singular. Keep it)
│   └── README.md
├── simulation_agent/
│   ├── __init__.py
│   ├── forecasting.py
│   ├── scenarios.py
│   ├── products.py
│   ├── narration.py
│   ├── report.py
│   └── README.md
├── recommendation_agent/
│   ├── __init__.py
│   ├── core.py
│   ├── contracts.py
│   ├── ml/  (features.py, priority_model.py, train_priority_model.py, model/priority_model.joblib)
│   └── tests/
└── coordinator_agent/
    ├── main.py
    ├── orchestrator.py
    ├── agent_stubs.py
    ├── data_access.py
    ├── contracts.py
    └── README.md                   (this file)
```

### Rules that make the merge safe
1. **Lowercase folder names with underscores only.** `Simulation_agent` and `spending-agent`
   break imports, and Windows hides case problems that fail on other systems.
2. **No per-agent copies of the data.** Delete any `output/` folder inside an agent's own folder.
   Everyone reads the root `output/`. If the data is regenerated, tell the whole team,
   because customer numbers change for everyone.
3. **One data generator.** Do not merge `data/generate_data.py` from the Simulation branch.
   Two generators create two different datasets.
4. **No `venv/` and no `__pycache__/` in any commit.**
5. **Import only the bare module names:** `from goal_agent import ...`,
   `from spending_agent_tools import ...`, `from shared.finance_utils import ...`,
   `from simulation_agent import ...`. Never `from goals_agent import ...`. That is the folder
   name, and it resolves to an empty module.

### Merge process
1. Open `main` on GitHub and compare its current layout with the tree above.
   If main's layout differs, fix it in one small restructure commit before merging
   anything else.
2. Each owner syncs main into their own branch first (`git fetch origin`, then
   `git merge origin/main`), fixes the layout, and checks that their agent still runs.
3. Merge order: shared data and `shared/`, then Spending, Goals, Simulation, then
   Recommendation, then Coordinator last, because the Coordinator imports all of them.
4. After merging, clone fresh into an empty folder, follow the setup section above, and
   confirm `http://127.0.0.1:8000/dashboard/CUSTE2483D` returns all four tabs without errors.

## Combined `requirements.txt` (to be added at the root)
```
pandas
requests
fastapi
uvicorn
scikit-learn
joblib
Faker
```
Pin exact versions once everyone runs `pip freeze` and the team agrees. In particular,
scikit-learn must match the version Alaa trained her saved model with.

## Notes for each owner

### Alia (Spending)
- `customer_id` format `CUSTXXXXXX` confirmed.
- The Coordinator calls both functions: `analyze_customer()` for the numbers and
  `generate_coach_message()` for the text.
- When `guardrail_warning` is set, the Coordinator logs it and shows a fallback sentence built
  from the numbers. You don't need your own fallback.
- Please change the default data paths in `spending_agent_tools.py` to the root folder:
  `Path(__file__).resolve().parent.parent / "output"`. The Coordinator currently overrides
  `CUSTOMERS_PATH` and `TRANSACTIONS_PATH` at import time.
- `call_llm()` only catches `ConnectionError`. A missing model raises
  `requests.exceptions.HTTPError`. The Coordinator catches it now, but please catch it in your code too.
- `MODEL` is hardcoded. Please make it configurable (for example an environment variable).

### Aya (Goals)
- The output is richer than the original placeholder: installment suggestions, gap, status.
- Your agent returns `{"error": ...}` instead of raising an exception, while Spending raises
  `SpendingAgentError`. The Coordinator handles both. The two agents still behave differently on failure.
- Please change the default data paths to the root `output/` folder, as for Spending.
- If `llama3.2` is not pulled, you get a 404 line in the terminal and `customer_message: null`.
  That is safe behavior.
- The folder is `goals_agent` but the file is `goal_agent.py`. This caused several import
  typos on our side. We keep it as it is. Always import `goal_agent`.

### Alaa (Simulation)
- Keep only the six package files in `simulation_agent/`. Move `run_simulation.py` to the root,
  and merge your `requirements.txt` into the root one.
- `load_data()` appears to have two versions: the file we integrated returns three values
  (transactions, goals, products), but newer code expects four (customers first).
  The Coordinator handles both. Please confirm which one is final.
- `forecasting.py` and `products.py` do not validate `customer_id`. An unknown ID raises a raw
  `IndexError`. The Coordinator catches it, but a clear error message from your side is better.
- `savings_rate` and the goal allocation logic should live in one shared function that Goals
  also uses (see Known issues below).
- Your agent uses `llama3`, which is a different model family from Spending and Goals.

### Ahlam (Recommendation)
- The Coordinator passes `None` for any agent that failed (it converts the error dicts first).
- Please do not merge `scripts/local_demo.py`. It assumes a different folder layout and will not run on main.
- Please state the exact scikit-learn version the model was trained with. If versions differ,
  `priority` becomes `null` and the rest still works.
- The Overview shows two different gap numbers (see Known issues below).

## Known issues (to resolve before Demo Day)
1. **Two definitions of "gap".** For the same customer and goal, Goals reports a monthly gap of
   3,675.40 (required minus disposable income after essentials), while Simulation reports a
   shortfall of 11,653.91 (required minus what the customer actually saves). Both are correct
   by their own definition, but the Overview shows both. The team must pick one definition or
   label them clearly.
2. **Rounding drift.** Spending reports average spending as `22557.92`, Simulation as
   `22557.923333...` for the same customer. The cause is that each agent calculates income
   and spend separately. A shared function would fix this.
3. **Three different Ollama models** across three agents. This makes setup harder and request
   time longer, because models are swapped in and out of memory.
4. **Inconsistent error patterns** (exceptions vs error dicts vs raw errors). Handled by the
   Coordinator for now.
5. **Slow requests** (1-2 minutes). The agents run one after another, with no caching.