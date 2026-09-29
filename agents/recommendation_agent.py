"""
Recommendation Agent
=====================
This is the fan-in step of the pipeline. It does NOT re-analyze
transactions, re-read goals.csv, or redo any math the other agents
already did. It only reads the dicts that Spending, Goals, and Simulation
already produced, and turns them into one clear, explainable
recommendation — using their actual numbers, never invented ones.

Design choices, on purpose (unchanged from the original version):
  - No scoring model, no ML, no LLM call. Every "why" in the output is a
    direct readout of a number one of the three agents already computed.
    That keeps it demo-safe (deterministic, testable, no hallucination
    risk) and matches "don't overengineer."
  - Any of the three inputs can be None (that agent failed, or wasn't
    needed for this question). The agent degrades gracefully rather than
    crashing, and is explicit in the output about which inputs it used.
  - Output shape follows the schema given in the project brief, plus a
    `goal_impact` block and a `used_inputs` block, because the UI needs
    both a "Goal Impact" section and a "Why am I seeing this?" section.

What changed vs. the original version:
  Goals and Simulation used to be small placeholders. They're now the
  real agents from the team's `goal_agent` and `Simulation_agent`
  branches, which return much richer output — the Goals Agent computes
  an actual disposable-income gap and concrete purchase-to-installment
  suggestions; the Simulation Agent computes per-goal feasibility over
  the customer's real saving history, ranked spending-cut scenarios, and
  ranked savings-product options. This file's job — combine whatever is
  available into one grounded recommendation, degrade gracefully when
  something is missing — hasn't changed; only the field names it reads
  have, plus two additions that make use of information that simply
  didn't exist before:
    - the Goals Agent's own installment suggestion is now preferred as
      the "how to close the gap" action, ahead of the older "redirect
      your overspending" phrasing, because it's a concrete, priced
      option rather than a generic instruction;
    - a savings-product recommendation, sourced from the Simulation
      Agent's ranked product options, is available for customers who
      have a healthy surplus but no gap to close.

What's new in this version:
  A small ML layer (agents/ml/) now scores each result with a
  data-driven "priority_score" (0-100) and "priority_label"
  (High/Medium/Low), using a RandomForestRegressor trained on real
  Spending/Goals/Simulation features across the customer base (see
  agents/ml/train_priority_model.py for exactly what it predicts, how
  its training labels were built, and why). This is purely additive: it
  does NOT decide which recommendations are shown or their order — that
  stays fully rule-based and explainable, for the same "no hallucination
  risk, always traceable to a real number" reasons as before. If the
  model file is missing or scikit-learn isn't installed, priority is
  simply omitted (None) and everything else works exactly as before.

Public contract (unchanged):
    run(customer_id, spending, goals, simulation) -> dict
"""

from agents.ml.features import extract_features
from agents.ml.priority_model import predict_priority


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
    primary_goal = goals.get("primary_goal") if goals else None
    if not primary_goal:
        return None

    reason_parts = [
        f"Your \"{primary_goal['goal_name']}\" goal needs about "
        f"{_fmt(primary_goal['monthly_required'])} per month to reach "
        f"{_fmt(primary_goal['target_amount'])}"
        + (f" in {primary_goal['duration_months']} months." if primary_goal.get("duration_months") is not None else ".")
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

    sim_goal = _find_sim_goal(primary_goal["goal_name"], simulation)
    months_current = None
    on_track = None
    if sim_goal:
        months_current = sim_goal.get("actual_months_needed")
        on_track = sim_goal.get("status") in ("ahead", "on track")
        planned = sim_goal.get("planned_months", primary_goal.get("duration_months"))
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
    primary_goal = goals.get("primary_goal") if goals else None
    if not primary_goal:
        return None

    sim_goal = _find_sim_goal(primary_goal["goal_name"], simulation)
    current_monthly_pace = sim_goal.get("allocated_monthly_savings") if sim_goal else None
    months_current = sim_goal.get("actual_months_needed") if sim_goal else None
    on_track = (sim_goal.get("status") in ("ahead", "on track")) if sim_goal else None

    return {
        "goal_name": primary_goal["goal_name"],
        "target_amount": primary_goal["target_amount"],
        "duration_months": primary_goal.get("duration_months"),
        "monthly_required": primary_goal["monthly_required"],
        "current_monthly_pace": current_monthly_pace,
        "projected_months_to_reach": months_current,
        "on_track": on_track,
    }


def run(customer_id: str, spending: dict = None, goals: dict = None, simulation: dict = None) -> dict:
    """
    The single entry point the Coordinator calls after Spending, Goals,
    and Simulation have run. Any of the three may be None.
    """
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
    if goals and goals.get("primary_goal"):
        summary_parts.append(f"Your top goal is \"{goals['primary_goal']['goal_name']}\".")
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
