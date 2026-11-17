"""
Recommendation Agent — public package API.

    from recommendation_agent import run_recommendation_agent
    recommendation = run_recommendation_agent(spending_output, goals_output, simulation_output)

See core.py for the implementation and README.md for how to wire this
into the Coordinator and how to run it locally.
"""
from .core import run, run_recommendation_agent

__all__ = ["run", "run_recommendation_agent"]
