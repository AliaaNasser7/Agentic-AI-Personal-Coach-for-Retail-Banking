"""
simulation_agent/errors.py

Matches the SpendingAgentError pattern used elsewhere in the project, so
the Coordinator can catch simulation-specific failures by type instead of
relying on a broad except clause.
"""


class SimulationAgentError(Exception):
    """Raised when the Simulation Agent can't compute a result for the given input."""
    pass
