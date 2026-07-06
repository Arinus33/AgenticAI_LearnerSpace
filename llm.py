"""Central place to create the LLM.

WHY THIS FILE EXISTS (API-call saving + failure handling):
1. MOCK MODE  -> set MOCK_MODE=1 in .env and the whole system runs with
   deterministic canned responses and ZERO API calls. Great for testing
   the multi-agent graph logic before spending any quota.
2. CACHING    -> LangChain's SQLiteCache stores every (prompt -> response)
   pair on disk. If you re-run the same request, no API call is made.
3. RETRIES    -> llm.with_retry() retries transient API failures
   (rate limits, network blips) up to 3 times with backoff.
"""

import os
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM

load_dotenv()

MOCK = os.getenv("MOCK_MODE", "0") == "1"

if not MOCK:
    # Failure handling + API saving: disk cache for repeated prompts.
    # (Disabled in mock mode so tests are always fresh.)
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache

    set_llm_cache(SQLiteCache(database_path=".llm_cache.db"))


class MockStudyLLM(LLM):
    """Deterministic offline stand-in for the real model (zero API calls).

    It inspects the incoming prompt to behave like a sensible supervisor
    or worker, so the full graph can be exercised end-to-end.
    """

    @property
    def _llm_type(self) -> str:
        return "mock-study-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Any = None, **kwargs: Any) -> str:
        p = prompt.lower()

        # --- supervisor routing prompt ---
        if "you are a supervisor" in p:
            wants = []
            # crude keyword intent detection on the student's request line
            request_line = next((l for l in p.splitlines()
                                 if l.startswith("student request:")), "")
            if any(w in request_line for w in ["summar", "notes", "revise"]):
                wants.append("summarizer")
            if any(w in request_line for w in ["quiz", "question", "test me"]):
                wants.append("quiz_master")
            if any(w in request_line for w in ["plan", "schedule", "timetable", "day"]):
                wants.append("planner")
            if not wants:
                wants = ["summarizer"]

            completed_line = next((l for l in p.splitlines()
                                   if l.startswith("agents already completed:")), "")
            for agent in wants:
                if agent not in completed_line:
                    return agent
            return "FINISH"

        # --- worker prompts ---
        if "quiz master" in p:
            return ("MOCK QUIZ:\n1) What does a PCB store? \n2) MCQ: RR with a huge "
                    "quantum behaves like (a) SJF (b) FCFS (c) Priority\n"
                    "Answer key: 1) process state/PID/registers  2) b")
        if "study planner" in p:
            return ("MOCK PLAN:\nDay 1: revise summary\nDay 2: attempt quiz\n"
                    "Day 3: full self-test")
        return ("MOCK SUMMARY:\n- Process = program in execution (PCB tracks it)\n"
                "- Scheduling: FCFS, SJF, RR, Priority (+aging)\n"
                "- Deadlock needs 4 conditions simultaneously")


def get_llm(temperature: float = 0.2):
    """Return a chat model. Groq (free tier) by default, mock if requested."""
    if MOCK:
        return MockStudyLLM()

    from langchain_groq import ChatGroq  # lazy import: mock mode needs no key

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=temperature,
    )
    # Failure handling: automatic retry on transient errors
    return llm.with_retry(stop_after_attempt=3)
