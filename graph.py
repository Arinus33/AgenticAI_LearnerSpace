"""LangGraph wiring — SUPERVISOR orchestration pattern.

        ┌────────────┐
        │ supervisor │◄──────────────┐
        └─────┬──────┘               │
   routes to one of...              │ (each worker reports back)
   ┌─────────┼───────────┐          │
   ▼         ▼           ▼          │
summarizer quiz_master planner ─────┘
        │
        ▼ (when supervisor says FINISH)
     assemble  ->  END

WHY SUPERVISOR (and not Pipeline/Parallel)?
The student's request decides WHICH agents are needed ("just a quiz" vs
"summary + plan"). That dynamic routing is exactly what the Supervisor
pattern is for. A Pipeline would waste API calls running every agent
every time.
"""

from langgraph.graph import StateGraph, END

from state import StudyState
from agents.supervisor import supervisor_node
from agents.workers import summarizer_node, quiz_master_node, planner_node


def assemble_node(state: StudyState) -> dict:
    """No LLM call here — plain Python stitches outputs together (free)."""
    parts = []
    if state.get("summary"):
        parts.append("## 📌 Summary\n" + state["summary"])
    if state.get("quiz"):
        parts.append("## ❓ Practice Quiz\n" + state["quiz"])
    if state.get("plan"):
        parts.append("## 🗓️ Study Plan\n" + state["plan"])
    final = "\n\n".join(parts) or "The supervisor decided no agent was needed."
    return {"final_answer": final}


def route(state: StudyState) -> str:
    return state["next_agent"]


def build_graph():
    g = StateGraph(StudyState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("summarizer", summarizer_node)
    g.add_node("quiz_master", quiz_master_node)
    g.add_node("planner", planner_node)
    g.add_node("assemble", assemble_node)

    g.set_entry_point("supervisor")

    # supervisor decides where to go next
    g.add_conditional_edges(
        "supervisor",
        route,
        {
            "summarizer": "summarizer",
            "quiz_master": "quiz_master",
            "planner": "planner",
            "FINISH": "assemble",
        },
    )

    # every worker reports back to the supervisor
    for worker in ["summarizer", "quiz_master", "planner"]:
        g.add_edge(worker, "supervisor")

    g.add_edge("assemble", END)
    return g.compile()
