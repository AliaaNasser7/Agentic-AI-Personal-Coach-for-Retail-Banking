"""
simulation_agent/narration.py

Step 6: turn the computed report into natural language for the customer.

Every number is built in Python first (build_fact_sentences) - the LLM, if
used, is only ever allowed to rephrase those already-correct sentences into
a flowing paragraph, never to compute or invent a number itself. This was a
deliberate fix after an earlier free-form-generation version was tested
against real output and found to hallucinate a month count, misread a
percentage, and invent a time horizon that wasn't in the data.
"""

import requests


def build_fact_sentences(report):
    sentences = []
    cf = report["cash_flow"]
    sentences.append(
        f"You're earning about {cf['avg_monthly_income']:,.0f} per month and spending about "
        f"{cf['avg_monthly_spend']:,.0f}, leaving a net of {cf['avg_monthly_net']:,.0f} per month."
    )

    bp = report["baseline_projection"]
    last = bp["projection"][-1]
    months_ahead = len(bp["projection"])
    sentences.append(
        f"Your balance is currently {bp['last_known_balance']:,.0f} as of {bp['last_known_date']}, "
        f"and at this pace it should reach about {last['projected_balance']:,.0f} by {last['date']}."
    )

    for g in report["goals"]["per_goal"]:
        months_txt = (
            f"about {g['actual_months_needed']:.0f} months"
            if g["actual_months_needed"] is not None
            else "an unclear amount of time at the current pace"
        )
        sentences.append(
            f"Your '{g['goal_name']}' goal is currently {g['status']}: at your current saving rate "
            f"it would take {months_txt}, versus the {g['planned_months']}-month plan."
        )

    overall = report["goals"]["overall"]
    if overall["status"] == "no_goals_set":
        sentences.append(
            f"You haven't set any savings goals yet. With about {cf['avg_monthly_net']:,.0f} left over "
            f"each month, you're in a good position to start one."
        )
    elif overall["status"] == "behind":
        sentences.append(
            f"Overall, you're saving {overall['avg_monthly_saved']:,.0f} per month toward your goals "
            f"but would need {overall['total_required_monthly']:,.0f} per month - "
            f"a shortfall of {overall['monthly_shortfall']:,.0f} per month."
        )
    else:
        sentences.append("Overall, you're on pace to meet your combined savings goals.")

    if report["top_scenarios"]:
        best = report["top_scenarios"][0]
        cat, pct = list(best["scenario"].items())[0] if len(best["scenario"]) == 1 else (None, None)
        if cat:
            pct_txt = f"{abs(pct) * 100:.0f}%"
            direction = "cutting" if pct < 0 else "increasing"
            sentences.append(
                f"Your best option to free up cash: {direction} {cat} spending by {pct_txt} would add "
                f"{best['monthly_gain']:,.0f} to your monthly net, reaching about "
                f"{best['projection'][-1]['projected_balance']:,.0f} in {months_ahead} months instead of "
                f"{last['projected_balance']:,.0f}."
            )
        else:
            sentences.append(
                f"Your best option to free up cash would add {best['monthly_gain']:,.0f} to your monthly net, "
                f"reaching about {best['projection'][-1]['projected_balance']:,.0f} in {months_ahead} months "
                f"instead of {last['projected_balance']:,.0f}."
            )

    flexible = report["top_products"]["flexible"]
    locked = report["top_products"]["locked"]
    if flexible:
        best_flex = flexible[0]
        sentences.append(
            f"For easy-access savings, the best option found is {best_flex['contribution_type']} contributions "
            f"of {best_flex['contribution_amount']:,.0f} into the {best_flex['product']}, earning about "
            f"{best_flex['interest_earned']:,.0f} in interest over {best_flex['months_simulated']} months, "
            f"with no lock-in period."
        )
    if locked:
        best_locked = locked[0]
        sentences.append(
            f"For a higher return with funds locked away, {best_locked['contribution_type']} contributions "
            f"of {best_locked['contribution_amount']:,.0f} into the {best_locked['product']} would earn about "
            f"{best_locked['interest_earned']:,.0f} in interest over {best_locked['months_simulated']} months."
        )

    return sentences


def narrate_report(report, model="llama3", host="http://localhost:11434", use_llm_polish=True):
    """
    Builds the guaranteed-correct fact sentences first. Then, optionally,
    asks the LLM ONLY to smooth them into a flowing paragraph - explicitly
    forbidding it from changing any number. If the LLM is unreachable, or
    you set use_llm_polish=False, the fact sentences are joined directly -
    still fully correct, just less stylistically fluid.
    """
    facts = build_fact_sentences(report)
    fallback = " ".join(facts)

    if not use_llm_polish:
        return fallback

    facts_block = "\n".join(f"- {s}" for s in facts)
    prompt = f"""You are a retail banking assistant. Below are factual sentences that are
ALREADY CORRECT and already contain every number the customer needs to see.

Your ONLY job is to combine them into one warm, flowing paragraph (3-5 sentences).

STRICT RULES:
- Do NOT change, round, recalculate, or rephrase any number, percentage, date, or month count.
- Every digit that appears below must appear identically in your output.
- You may only change connecting words, sentence order, and tone.
- Do not add any number, fact, or claim that is not already listed below.

FACTS:
{facts_block}

Write the paragraph now:"""

    try:
        response = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return fallback
    except Exception:
        return fallback
