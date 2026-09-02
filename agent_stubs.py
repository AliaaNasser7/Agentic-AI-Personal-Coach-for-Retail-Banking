from coordinator_agent.contarcts import EXAMPLE_COACH_MESSAGE, EXAMPLE_SPENDING_ANALYSIS


class SpendingAgentError(Exception):
    """Stub placeholder — matches the real exception from spending_agent_tools.py.
    When the real file arrives, just change the import, this class name stays the same."""
    pass


def analyze_customer(customer_id):
    # STUB — will become: from spending_agent_tools import analyze_customer
    if not customer_id:
        raise SpendingAgentError("customer_id is required")
    return EXAMPLE_SPENDING_ANALYSIS


def generate_coach_message(customer_id):
    # STUB — will become: from spending_agent_local import generate_coach_message
    return EXAMPLE_COACH_MESSAGE
