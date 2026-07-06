"""SUPERVISOR agent — the 'manager' in the Supervisor orchestration pattern.

It looks at the student's request + what has already been done, and decides
which worker agent should act next, or FINISH.

Failure handling #3 lives here:
- routing VALIDATION: if the LLM returns garbage, we fall back to a safe
  default instead of crashing.
- LOOP CAP: after MAX_STEPS the supervisor force-finishes so a confused
  LLM can never loop forever (and burn your API quota).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm import get_llm
from state import StudyState

VALID_ROUTES = {"summarizer", "quiz_master", "planner", "FINISH"}
MAX_STEPS = 6

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are a supervisor managing three study-helper agents:
- summarizer: condenses the student's notes into key points
- quiz_master: creates practice questions from the notes
- planner: builds a day-by-day revision timetable

Student request: {user_request}
Agents already completed: {completed}

Decide who should work NEXT. Only pick an agent if the request needs it and
it has NOT already run. If everything the student asked for is done, answer FINISH.

Reply with EXACTLY one word: summarizer, quiz_master, planner, or FINISH."""
)


def supervisor_node(state: StudyState) -> dict:
    steps = state.get("steps_taken", 0)

    # Failure handling: hard cap on loop length
    if steps >= MAX_STEPS:
        return {"next_agent": "FINISH", "steps_taken": steps}

    chain = ROUTER_PROMPT | get_llm(temperature=0) | StrOutputParser()
    raw = chain.invoke(
        {
            "user_request": state["user_request"],
            "completed": ", ".join(state.get("completed", [])) or "none",
        }
    ).strip()

    # Failure handling: validate the route, never trust raw LLM output
    route = next((r for r in VALID_ROUTES if r.lower() in raw.lower()), None)
    if route is None or route in state.get("completed", []):
        route = "FINISH"

    return {"next_agent": route, "steps_taken": steps + 1}
