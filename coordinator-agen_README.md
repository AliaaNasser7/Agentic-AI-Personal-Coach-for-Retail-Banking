# Coordinator Agent

## Role
Orchestrates the specialist agents (Spending, Goals, Simulation,
Recommendation), owns customer identification, and defines the shared
data contracts each agent must follow.

## Files
| File | Purpose |
|---|---|
| `data_access.py` | Loads `customers.csv` and looks up a customer by ID. |
| `contracts.py` | The exact output shape each agent must return. |
| `agent_stubs.py` | Stand-in versions of each agent, matching the real contract shapes, used until real agents are ready. |
| `orchestrator.py` | `load_dashboard(customer_id)` — calls all agents and combines results into one structure for the dashboard tabs. |

## Status
- Coordinator orchestration logic: working, tested with stubs.
- Spending Agent: real handoff received from Alia, contract updated to match (`analyze_customer`, `generate_coach_message`).
- Goals Agent: not yet received — using a stub.
- Simulation Agent: not yet received — using a stub.
- Recommendation Agent: not yet received — using a stub.

## Q&A with teammates

**From Alia (Spending Agent) — her questions:**

1. **Will `customer_id` reach you in the same format as the dataset (`CUSTXXXXXX`)?**
   → Yes, confirmed. No change needed.

2. **Should the Coordinator call `analyze_customer()` only, or `generate_coach_message()` too?**
   → Both. Raw numbers go into shared state (Simulation needs them too).
   The coach message becomes the Spending tab's own text. The Overview
   tab's combined message is generated separately by the Recommendation
   Agent from the raw numbers, not from an already-summarized message.

3. **If `guardrail_warning` fires, should Spending Agent return something
   different, or does the Coordinator decide?**
   → Coordinator decides. If it fires, Coordinator logs it and shows a
   fallback sentence built directly from the numeric analysis instead
   of the flagged message. Spending Agent doesn't need its own fallback.

## Open questions (not yet resolved)
- File path convention across agents — needs team alignment (Spending
  Agent's scripts expect data in the same folder as the scripts; my
  code expects a shared `output/` folder at the repo root).