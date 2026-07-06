"""Shared state that flows through the LangGraph graph.

DESIGN DECISION (shared vs isolated state):
We use ONE shared state object (a TypedDict) that every agent can read,
but each agent only WRITES to its own output field. This gives us the
simplicity of shared state without agents overwriting each other's work.
"""

from typing import TypedDict, List, Optional


class StudyState(TypedDict, total=False):
    # --- inputs ---
    user_request: str          # what the student asked for
    notes_text: str            # raw text of the student's notes file
    retrieved_chunks: List[str]  # relevant chunks picked by the local retriever

    # --- routing / control ---
    next_agent: str            # supervisor's routing decision
    steps_taken: int           # loop-safety counter (failure handling #3)
    completed: List[str]       # which agents have already run

    # --- agent outputs (each agent writes ONLY its own field) ---
    summary: Optional[str]     # written by summarizer agent
    quiz: Optional[str]        # written by quiz_master agent
    plan: Optional[str]        # written by planner agent

    # --- final ---
    final_answer: str
