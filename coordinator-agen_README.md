``

## Ollama / model setup — each agent currently uses a different model

```powershell
ollama pull llama3.1:8b   # Spending Agent
ollama pull llama3.2      # Goals Agent
ollama pull llama3        # Simulation Agent
```
All three can coexist locally — no conflict, just three separate downloads.
**Open discussion for the team:** worth standardizing on one model
eventually to simplify setup for anyone re-cloning this repo, but not
blocking right now.

## Known GPU/driver issue (if you hit it)

If `ollama run <model>` fails with a CUDA error like:
CUDA error: the provided PTX was compiled with an unsupported toolchain

This is a known Ollama/older-NVIDIA-driver incompatibility, not a code bug.
Ollama requires driver **570+** for GPUs with compute capability 5.0–6.2
(covers many older/entry-level NVIDIA laptop GPUs, e.g. MX-series).

Fixes, in order of preference:
1. Update the NVIDIA driver to 570+ (check `nvidia-smi` for your current
   version), clean install, restart.
2. If you want a GPU-free fallback: `setx OLLAMA_NUM_GPU 0`, then **fully
   quit and restart the Ollama background service** (right-click its tray
   icon → Quit, then reopen) — setting it with `$env:` in one terminal only
   affects new processes from that terminal, not an already-running Ollama
   service.



## Run-time behavior worth knowing

- **A single dashboard request can take 1-2 minutes.** It calls 3 separate
  local LLMs in sequence (Spending → Goals → Simulation, Recommendation once
  it's real too). This is expected on modest hardware, not a bug.
- **If an LLM call fails or times out, the dashboard still loads.** Spending
  falls back to a plain, numbers-based sentence; Goals/Simulation skip the
  LLM narration and return `null`/fact-only text instead of crashing.

## Notes & open items for each agent

### To Alia (Spending Agent)
- Confirmed: `customer_id` format matches (`CUSTXXXXXX`).
- Coordinator calls both `analyze_customer()` (raw numbers, also useful to
  Simulation) and `generate_coach_message()` (tab text).
- `guardrail_warning` handling: Coordinator owns this — logs it and shows a
  fallback sentence built from the raw numbers instead of the flagged message.
  You don't need your own fallback.
- Minor: `call_llm()` only catches `ConnectionError` — a wrong/unavailable
  model name raises `requests.exceptions.HTTPError` instead, which wasn't
  originally caught. Coordinator now catches both, but worth catching on
  your end too.
- Your `spending_agent_tools.py` computes its own `CUSTOMERS_PATH`/
  `TRANSACTIONS_PATH` relative to its own file location — Coordinator
  overrides both at import time to point at the shared `/output` folder.

### To Aya (Goals Agent)
- Real output shape is richer than our original placeholder — includes
  installment-based suggestions, not just an on-track/behind flag.
- Your agent returns `{"error": ...}` as a plain dict rather than raising an
  exception — Coordinator handles both this pattern and Alia's exception
  pattern, but worth knowing these two agents currently behave differently
  on failure.
- `use_llm_message=True` — confirmed working once `llama3.2` is pulled;
  fails silently (prints, returns `null`) if the model's missing, which is
  good, safe behavior.
- Minor naming inconsistency: folder is `goals_agent` (plural), file inside
  is `goal_agent.py` (singular) — caused a few import typos on our end,
  not urgent to rename, just flagging.

### To Alaa (Simulation Agent)
- Fully wired in — genuinely the richest, most complete output of the three
  real agents so far.
- Per your own README's open item: **`savings_rate`/proportional allocation
  logic should probably move to one shared function** both you and Aya call
  — right now Spending, Goals, and Simulation each compute
  income/spending/savings slightly independently, and we've already seen
  minor rounding drift (e.g. `22557.92` vs `22557.923333...` for the same
  customer's average spend, computed by two different agents from the same
  data). Not breaking anything today, worth resolving before the demo.
- `forecasting.py`/`products.py` don't validate `customer_id` — an unknown
  ID raises a raw `IndexError` rather than a clean message. Coordinator
  catches this broadly for now; a proper check on your end (like Alia's
  `SpendingAgentError` pattern) would be cleaner.
- Confirmed: Coordinator calls `build_simulation_output()`, not
  `run_full_simulation_report()`.
- Model used: `llama3` (different from Spending's `llama3.1:8b` and Goals'
  `llama3.2` — see model setup above).

### To Ahlam (Recommendation Agent) — not built yet
- This is the last piece. It should receive the **real** outputs of
  Spending, Goals, and Simulation (all working now) and synthesize them
  into one Overview-tab summary.
- Suggestion: don't re-summarize Spending's already-summarized
  `coach_message` — use the raw numeric fields instead, same principle
  Coordinator already follows for the other agents.

## Known limitations
- No shared "savings rate" calculation yet — see note to Ahlam above.
- Recommendation Agent is still a stub (`EXAMPLE_RECOMMENDATION_OUTPUT`).
- Three different local models in use across agents (see model setup above).
- A single dashboard request is slow (1-2 min) due to sequential local LLM
  calls — no caching or parallelization yet.