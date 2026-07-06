"""The three WORKER agents. Each one:
1. Uses the local retriever to pull only the relevant chunks of the notes
   (keeps prompts short -> saves tokens/API cost).
2. Makes exactly ONE LLM call.
3. Writes ONLY to its own field in the shared state, then marks itself done.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm import get_llm
from state import StudyState
from tools.retriever import retrieve


def _context(state: StudyState) -> str:
    chunks = retrieve(state["user_request"], state.get("notes_text", ""))
    return "\n---\n".join(chunks) if chunks else "(no notes provided)"


def _run(prompt_text: str, state: StudyState) -> str:
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke(
        {"request": state["user_request"], "context": _context(state)}
    )


def summarizer_node(state: StudyState) -> dict:
    out = _run(
        """You are a study summarizer for a college student.
Student request: {request}
Relevant notes:
{context}

Write a crisp summary as 5-7 bullet points a student can revise from.""",
        state,
    )
    return {"summary": out, "completed": state.get("completed", []) + ["summarizer"]}


def quiz_master_node(state: StudyState) -> dict:
    out = _run(
        """You are a quiz master for a college student.
Student request: {request}
Relevant notes:
{context}

Create 5 practice questions (mix of MCQ and short answer) WITH an answer
key at the end.""",
        state,
    )
    return {"quiz": out, "completed": state.get("completed", []) + ["quiz_master"]}


def planner_node(state: StudyState) -> dict:
    out = _run(
        """You are a study planner for a college student.
Student request: {request}
Relevant notes:
{context}

Build a realistic day-by-day revision plan (max 5 days, ~2 hours/day)
covering these topics, ending with a self-test day.""",
        state,
    )
    return {"plan": out, "completed": state.get("completed", []) + ["planner"]}
