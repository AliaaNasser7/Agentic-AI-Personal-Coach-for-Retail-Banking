"""
Recommendation Agent — core logic
====================================
This is the fan-in step of the pipeline. It does NOT re-analyze
transactions, re-read goals.csv, or redo any math the other agents
already did. It only reads the dicts that Spending, Goals, and
Simulation already produced, and turns them into one clear, explainable
recommendation — using their actual numbers, never invented ones.

Design choices, on purpose:
  - Rule-based content generation, not an LLM call. Every "why" in the
    output is a direct readout of a number one of the three agents
    already computed. That keeps it demo-safe (deterministic, testable,
    no hallucination risk).
  - A small additive ML layer (see ./ml/) scores the overall result with
    a data-driven "priority" (0-100 + High/Medium/Low), but never
    decides which recommendations are shown or their order — see
    ./ml/train_priority_model.py for exactly what it predicts and why.
  - Any of the three inputs can be None (that agent failed, or wasn't
    needed for this question). The agent degrades gracefully rather
    than crashing, and is explicit in the output about which inputs it
    used (`used_inputs`).
  - Output includes a `goal_impact` block and a `used_inputs` block
    because the dashboard needs both a "Goal Impact" section and a
    "Why am I seeing this?" section.

Reads (see contracts.py for exact field-level shapes, matching the real
goal_agent / spending-agent / Simulation_agent branches):
  - spending: Spending Agent's analyze_customer() output (+ coach_message, ignored here)
  - goals:    Goals Agent's GoalAgent.recommend() output
  - simulation: Simulation Agent's build_simulation_output() output
Any of the three may be None.

Two entry points (see bottom of this file / __init__.py):
  - run_recommendation_agent(spending, goals, simulation) -> dict
        Exact signature the Coordinator's agent_stubs.py stub already
        expects — this is what gets wired in.
  - run(customer_id, spending, goals, simulation) -> dict
        Same thing, with an explicit customer_id for standalone/test use.
"""

from .ml.features import extract_features
from .ml.priority_model import predict_priority


def _fmt(n):
    """Formats a number for a user-facing sentence: no trailing .0 noise,
    thousands separator, always 0-2 decimals."""
    if n is None:
        return "N/A"
    return f"{n:,.0f}" if abs(n - round(n)) < 0.005 else f"{n:,.2f}"


def _find_sim_goal(goal_name, simulation):
    """Looks up the Simulation Agent's per-goal feasibility entry for a
    given goal name (both agents read the same goals.csv, so goal_name is
    a reliable join key). Returns None if simulation is missing or the
    goal isn't in its per-goal list (e.g. the customer has no goals)."""
    if not simulation or not goal_name:
        return None
    for g in simulation.get("goals", {}).get("per_goal", []):
        if g.get("goal_name") == goal_name:
            return g
    return None


def _build_spending_recommendation(spending):
    """One recommendation derived purely from Spending Agent output.
    Unchanged from the original version."""
    if not spending or not spending.get("overspending_categories"):
        return None

    top = spending["overspending_categories"][0]
    excess = round(top["month_amount"] - top["avg_other_months"], 2)
    if excess <= 0:
        return None

    return {
        "title": f"Reduce {top['category']} Spending",
        "reason": (
            f"In {top['month']}, your {top['category']} spending was "
            f"{_fmt(top['vs_avg_pct'])}% higher than your usual average "
            f"({_fmt(top['month_amount'])} vs. a typical {_fmt(top['avg_other_months'])})."
        ),
        "estimated_impact": f"Potential saving of about {_fmt(excess)} per month.",
        "action": f"Review your {top['category']} transactions for {top['month']} and cut back toward your usual average.",
        "_excess": excess,
        "_category": top["category"],
    }


def _pick_primary_goal(goals: dict):
    """The real GoalAgent.recommend() output lists every goal the
    customer has, fully analyzed, but doesn't itself pick one to lead
    with -- deciding that is a fan-in judgment call, which is exactly
    this agent's job. Rule: lead with whichever goal most needs
    attention right now (status == "needs_action", largest monthly_gap
    first); if every goal is already on track, lead with the first one.
    Simple and explainable, not a scoring model."""
    if not goals:
        return None
    goal_list = goals.get("goals") or []
    if not goal_list:
        return None
    needs_action = [g for g in goal_list if g.get("status") == "needs_action"]
    pool = needs_action or goal_list
    return max(pool, key=lambda g: g.get("monthly_gap") or 0)


def _build_goal_recommendation(goals, simulation):
    """One recommendation tying the customer's real financial position to
    their primary goal, using:
      - the Goals Agent's own disposable-income gap for THIS month, and
        its priced installment suggestion for closing that gap (the most
        concrete, specific action available), and
      - the Simulation Agent's longer-horizon feasibility read (based on
        actual saving history, not just this month) as a second signal,
        plus its best ranked spending-cut scenario as a fallback action
        when there's no installment suggestion to offer.
    """
    primary_goal = _pick_primary_goal(goals)
    if not primary_goal:
        return None

    # duration_months isn't part of the Goals Agent's own output (it
    # only tracks target/monthly_required) -- the Simulation Agent's
    # per-goal planned_months is the same concept, computed from the
    # same goals.csv, so use it here when available.
    sim_goal = _find_sim_goal(primary_goal["goal_name"], simulation)
    duration = sim_goal.get("planned_months") if sim_goal else None

    reason_parts = [
        f"Your \"{primary_goal['goal_name']}\" goal needs about "
        f"{_fmt(primary_goal['monthly_required'])} per month to reach "
        f"{_fmt(primary_goal['target_amount'])}"
        + (f" in {duration} months." if duration is not None else ".")
    ]

    gap = primary_goal.get("monthly_gap")
    if primary_goal.get("status") == "needs_action" and gap is not None and gap > 0:
        reason_parts.append(
            f"After essential expenses this month, you have about "
            f"{_fmt(primary_goal.get('disposable_income'))} available toward it — "
            f"a gap of {_fmt(gap)} per month."
        )
    elif primary_goal.get("status") == "on_track":
        reason_parts.append(
            "Your disposable income this month already covers what this goal needs."
        )

    months_current = None
    on_track = None
    if sim_goal:
        months_current = sim_goal.get("actual_months_needed")
        on_track = sim_goal.get("status") in ("ahead", "on track")
        planned = sim_goal.get("planned_months", duration)
        if months_current is not None:
            if on_track:
                reason_parts.append(
                    f"Based on your actual saving history, you're on track to reach it in "
                    f"about {months_current:.0f} months."
                )
            else:
                reason_parts.append(
                    f"Based on your actual saving history, it would take about "
                    f"{months_current:.0f} months — longer than the {planned}-month plan."
                )
        else:
            reason_parts.append(
                "Based on your actual saving history, this goal isn't reachable at the current pace."
            )

    # Prefer the Goals Agent's own priced installment suggestion — a
    # concrete, specific action — over a generic instruction.
    suggestions = primary_goal.get("suggestions") or []
    top_suggestion = suggestions[0] if suggestions else None

    impact = "N/A"
    action = f"Keep contributing toward your \"{primary_goal['goal_name']}\" goal."

    if top_suggestion:
        impact = (
            f"Frees up about {_fmt(top_suggestion['monthly_cash_freed'])} per month "
            f"({_fmt(top_suggestion['contribution_to_goal_pct'])}% of what this goal requires)."
        )
        action = (
            f"Convert \"{top_suggestion['description']}\" "
            f"({_fmt(top_suggestion['original_amount'])}, {top_suggestion['category']}) into a "
            f"{top_suggestion['tenor_months']}-month installment plan, and redirect the freed-up "
            f"cash toward your \"{primary_goal['goal_name']}\" goal."
        )
    elif simulation and simulation.get("top_scenarios"):
        best = simulation["top_scenarios"][0]
        scenario_items = list(best.get("scenario", {}).items())
        if scenario_items:
            category, pct = scenario_items[0]
            pct_txt = f"{abs(pct) * 100:.0f}%"
            impact = f"Adds about {_fmt(best['monthly_gain'])} to your monthly net."
            action = (
                f"Cut {category} spending by {pct_txt} and redirect the "
                f"{_fmt(best['monthly_gain'])}/month toward your \"{primary_goal['goal_name']}\" goal."
            )
    elif months_current is not None:
        impact = f"About {months_current:.0f} months to reach this goal at your current saving pace."
    elif primary_goal.get("disposable_income") is not None:
        action = (
            f"Set up an automatic monthly transfer of your {_fmt(primary_goal['disposable_income'])} "
            f"disposable income toward your \"{primary_goal['goal_name']}\" goal."
        )

    return {
        "title": f"Accelerate Your \"{primary_goal['goal_name']}\" Goal",
        "reason": " ".join(reason_parts),
        "estimated_impact": impact,
        "action": action,
        "_goal_name": primary_goal["goal_name"],
    }


def _build_savings_product_recommendation(simulation):
    """A recommendation sourced from the Simulation Agent's ranked
    savings/investment product options — useful for a customer with a
    healthy surplus but no pressing goal gap to close (the old version
    had nothing to offer in this case beyond a generic "keep saving").
    Prefers a flexible (no lock-in) option since it asks nothing of the
    customer they didn't already have; falls back to the best locked
    option if that's all that's available."""
    if not simulation:
        return None
    top_products = simulation.get("top_products") or {}
    flexible = top_products.get("flexible") or []
    locked = top_products.get("locked") or []
    best = (flexible or locked)[:1]
    if not best:
        return None
    product = best[0]
    is_flexible = bool(flexible)

    return {
        "title": f"Put Your Surplus to Work in {product['product']}",
        "reason": (
            f"Simulating {product['contribution_type']} contributions of "
            f"{_fmt(product['contribution_amount'])} into the {product['product']} "
            f"(annual rate {product['annual_rate'] * 100:.1f}%) over {product['months_simulated']} months."
            + ("" if is_flexible else f" {product.get('warning', '')}")
        ),
        "estimated_impact": f"Earns about {_fmt(product['interest_earned'])} in interest over {product['months_simulated']} months.",
        "action": (
            f"Move your {product['contribution_type'].replace('_', ' ')} contribution into the "
            f"{product['product']}{'' if is_flexible else ' (funds are locked for the product term)'}."
        ),
    }


def _general_savings_recommendation(spending):
    """Used when there's no overspending flag, no goal, and no product
    option — a safe, honest fallback instead of inventing a concern.
    Unchanged from the original version."""
    if not spending:
        return None
    surplus = spending.get("monthly_surplus")
    if surplus is None:
        return None
    if surplus > 0:
        return {
            "title": "Keep Building Your Surplus",
            "reason": f"Your spending looks consistent with your usual habits, and you have a monthly surplus of {_fmt(surplus)}.",
            "estimated_impact": f"Saving this surplus consistently adds up to about {_fmt(surplus * 12)} per year.",
            "action": "Consider setting up an automatic transfer of your surplus into a savings product.",
        }
    return {
        "title": "Address Your Monthly Deficit",
        "reason": f"Your average spending is currently higher than your average income, leaving a shortfall of {_fmt(abs(surplus))} per month.",
        "estimated_impact": "Closing this gap is the first step before any savings goal becomes realistic.",
        "action": "Review your largest spending categories this month and look for one or two to cut back on.",
    }


def _build_goal_impact_block(goals, simulation):
    """Feeds the UI's 'Goal Impact' section (current vs target, progress).
    current_monthly_pace and projected_months_to_reach now come from the
    Simulation Agent's per-goal feasibility (based on real saving
    history) rather than a single surplus number, and on_track reflects
    its "ahead"/"on track"/"behind" status."""
    primary_goal = _pick_primary_goal(goals)
    if not primary_goal:
        return None

    sim_goal = _find_sim_goal(primary_goal["goal_name"], simulation)
    current_monthly_pace = sim_goal.get("allocated_monthly_savings") if sim_goal else None
    months_current = sim_goal.get("actual_months_needed") if sim_goal else None
    on_track = (sim_goal.get("status") in ("ahead", "on track")) if sim_goal else None

    return {
        "goal_name": primary_goal["goal_name"],
        "target_amount": primary_goal["target_amount"],
        "duration_months": sim_goal.get("planned_months") if sim_goal else None,
        "monthly_required": primary_goal["monthly_required"],
        "current_monthly_pace": current_monthly_pace,
        "projected_months_to_reach": months_current,
        "on_track": on_track,
    }


def _infer_customer_id(spending, goals, simulation):
    """customer_id isn't passed in separately by the Coordinator's
    run_recommendation_agent(spending, goals, simulation) call — every
    one of the three agents already stamps it on their own output, so
    pull it from whichever is available instead of asking for it again."""
    for d in (spending, goals, simulation):
        if d and d.get("customer_id"):
            return d["customer_id"]
    return None


def run(customer_id: str = None, spending: dict = None, goals: dict = None, simulation: dict = None) -> dict:
    """
    Runs the Recommendation Agent. Any of spending/goals/simulation may
    be None (that agent failed, or wasn't needed). customer_id is
    optional -- if not given, it's read off whichever input dict has it.
    """
    if customer_id is None:
        customer_id = _infer_customer_id(spending, goals, simulation)

    if spending is None and goals is None and simulation is None:
        return {
            "customer_id": customer_id,
            "summary": "Not enough information is available to make a recommendation right now.",
            "recommendations": [],
            "goal_impact": None,
            "priority": None,
            "used_inputs": {"spending": False, "goals": False, "simulation": False},
            "analyzed_by": "Recommendation Agent",
        }

    recommendations = []

    goal_rec = _build_goal_recommendation(goals, simulation)
    spending_rec = _build_spending_recommendation(spending)

    def _clean(rec):
        return {k: v for k, v in rec.items() if not k.startswith("_")}

    if spending_rec and goal_rec:
        # Combine rather than just listing both separately, per the brief's example:
        # tie the overspending category directly to the goal it could fund.
        recommendations.append(_clean(goal_rec))
        # Only add the raw spending-only recommendation as a secondary item
        # if it points at a *different* category than the goal rec already covered.
        if spending_rec.get("_category") and goal_rec.get("action", "").find(spending_rec["_category"]) == -1:
            recommendations.append(_clean(spending_rec))
    elif goal_rec:
        recommendations.append(_clean(goal_rec))
    elif spending_rec:
        recommendations.append(_clean(spending_rec))
    else:
        fallback = _build_savings_product_recommendation(simulation) or _general_savings_recommendation(spending)
        if fallback:
            recommendations.append(fallback)

    if not recommendations:
        return {
            "customer_id": customer_id,
            "summary": "Not enough information is available to make a recommendation right now.",
            "recommendations": [],
            "goal_impact": None,
            "priority": None,
            "used_inputs": {
                "spending": spending is not None,
                "goals": goals is not None,
                "simulation": simulation is not None,
            },
            "analyzed_by": "Recommendation Agent",
        }

    # Summary: one or two sentences grounding the whole answer in real numbers.
    summary_parts = []
    if spending and spending.get("monthly_surplus") is not None:
        surplus = spending["monthly_surplus"]
        if surplus >= 0:
            summary_parts.append(f"You currently have a monthly surplus of {_fmt(surplus)}.")
        else:
            summary_parts.append(f"You currently have a monthly shortfall of {_fmt(abs(surplus))}.")
    primary_for_summary = _pick_primary_goal(goals)
    if primary_for_summary:
        summary_parts.append(f"Your top goal is \"{primary_for_summary['goal_name']}\".")
    if simulation and simulation.get("goals", {}).get("overall", {}).get("status") == "behind":
        shortfall = simulation["goals"]["overall"].get("monthly_shortfall")
        if shortfall:
            summary_parts.append(f"Overall, you're saving {_fmt(shortfall)} less per month than your goals need.")
    summary = " ".join(summary_parts) if summary_parts else "Here's a recommendation based on your available data."

    features = extract_features(spending, goals, simulation)
    priority = predict_priority(features)
    if priority:
        if priority["label"] == "High":
            summary += " This needs attention soon."
        elif priority["label"] == "Low":
            summary += " Your situation looks healthy overall."

    return {
        "customer_id": customer_id,
        "summary": summary,
        "recommendations": recommendations,
        "goal_impact": _build_goal_impact_block(goals, simulation),
        "priority": priority,
        "used_inputs": {
            "spending": spending is not None,
            "goals": goals is not None,
            "simulation": simulation is not None,
        },
        "analyzed_by": "Recommendation Agent",
    }


def run_recommendation_agent(spending_output: dict, goals_output: dict, simulation_output: dict) -> dict:
    """
    Drop-in replacement for the Coordinator's current stub in
    agent_stubs.py:

        def run_recommendation_agent(spending_output, goals_output, simulation_output):
            return EXAMPLE_RECOMMENDATION_OUTPUT

    Same name, same three positional args, same calling convention
    orchestrator.py already uses -- swap the stub's `return
    EXAMPLE_RECOMMENDATION_OUTPUT` for `return run_recommendation_agent(...)`
    (see this package's README for the exact one-line wiring change).
    """
    return run(spending=spending_output, goals=goals_output, simulation=simulation_output)
