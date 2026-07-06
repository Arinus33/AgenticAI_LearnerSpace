# 📚 StudySquad — a Multi-Agent Exam Prep Assistant

**Capstone project — Multi-Agent Systems Week**

StudySquad is a small team of AI agents that helps a student prepare for exams from their own class notes. You give it a `.txt` file of notes and a request in plain English (e.g. *"summarize my notes and quiz me"*), and a **Supervisor agent** decides which specialist agents need to run:

| Agent | Responsibility |
|---|---|
| 🧑‍💼 **Supervisor** | Reads the request, routes work to the right specialist, decides when the job is done |
| 📌 **Summarizer** | Condenses notes into revision-ready bullet points |
| ❓ **Quiz Master** | Generates practice questions + answer key from the notes |
| 🗓️ **Planner** | Builds a day-by-day revision timetable |

The agents coordinate through the **Supervisor orchestration pattern** built with **LangGraph**, so if you only ask for a quiz, only the Quiz Master runs — no wasted API calls.

---

## 🚀 How to run this on your laptop

Works on Windows, macOS, and Linux. You need **Python 3.10+**.

**1. Clone and enter the repo**
```bash
git clone https://github.com/<your-username>/study-squad.git
cd study-squad
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

**3. Set up your API key (free)**
```bash
# copy the example env file
cp .env.example .env        # Windows: copy .env.example .env
```
Then open `.env` and paste your key from https://console.groq.com — **Groq's free tier needs no credit card**, which is why this project uses it.

> 💡 **No API key? No problem.** Set `MOCK_MODE=1` in `.env` and the entire multi-agent flow runs with canned responses and **zero API calls** — perfect for testing the graph logic.

**4. Run it**
```bash
# Full study pack from the included sample notes (Operating Systems)
python main.py --request "summarize my notes, quiz me, and give me a study plan"

# Just a quiz (only 2 LLM calls total!)
python main.py --request "quiz me on scheduling algorithms"

# Your own notes
python main.py --notes path/to/my_notes.txt --request "make me a 3 day plan"
```

You'll see the combined output plus a line showing exactly which agents ran.

---

## 🏗️ Architecture — Supervisor pattern

```
                 ┌────────────┐
      ┌─────────►│ SUPERVISOR │──────────┐
      │          └─────┬──────┘          │ "FINISH"
      │   routes to one of...            ▼
      │   ┌──────────┬───────────┐   ┌──────────┐
      │   ▼          ▼           ▼   │ assemble │──► END
      │ Summarizer QuizMaster Planner└──────────┘
      │   │          │           │
      └───┴──────────┴───────────┘
          (each worker reports back to the supervisor)
```

**Why Supervisor and not Pipeline/Parallel?** The user's request decides *which* agents are needed. A Pipeline would run every agent every time (wasting API calls); Parallel+Aggregator would too. The Supervisor gives dynamic, request-driven routing — only the agents you actually need get invoked.

**Shared vs isolated state (decided before coding):** one shared `TypedDict` state flows through the graph, but each agent *writes only to its own field* (`summary`, `quiz`, `plan`). This gives the readability of shared state without agents clobbering each other — a middle ground between fully shared and fully isolated.

---

## 🧰 What I used, where, and why

| Technology | Where in the code | Why |
|---|---|---|
| **LangGraph** (`StateGraph`, conditional edges) | `graph.py` | The core orchestration layer. Conditional edges implement the supervisor's routing; normal edges send every worker back to the supervisor. This is exactly the Supervisor pattern from the course theory. |
| **LangChain LCEL chains** (`prompt \| llm \| parser`) | `agents/supervisor.py`, `agents/workers.py` | Clean, composable way to build each agent as prompt → model → output parser. |
| **LangChain `ChatPromptTemplate`** | both agent files | Keeps prompts versionable and separates prompt text from logic. |
| **LangChain `SQLiteCache`** | `llm.py` | **API-call saving:** identical prompts are answered from a local disk cache — re-running a test costs 0 calls. |
| **`with_retry()`** | `llm.py` | **Failure handling:** transient API errors (rate limits, network) auto-retry up to 3×. |
| **`FakeListLLM`** (mock mode) | `llm.py` | **API-call saving:** test the entire multi-agent flow with zero real calls. |
| **Groq free tier** (`langchain-groq`) | `llm.py` | Free, fast inference — realistic for a student budget. Swappable in one file. |
| **Lightweight local retriever (mini-RAG)** | `tools/retriever.py` | Retrieval idea from Week 2 (RAG), but done with keyword-overlap scoring instead of embeddings, so it costs **zero API calls** and keeps prompts short (fewer tokens per call). |
| **LangGraph shared state** (`TypedDict`) | `state.py` | Explicit contract for what flows between agents. |

## 🛡️ Failure-handling mechanisms (3 of them)

1. **Automatic retries** — `llm.with_retry(stop_after_attempt=3)` in `llm.py` handles flaky API calls.
2. **Route validation** — in `agents/supervisor.py`, the supervisor's raw LLM output is validated against a whitelist; garbage output falls back to `FINISH` instead of crashing, and an agent is never re-routed to twice.
3. **Loop cap** — `MAX_STEPS = 6` in the supervisor guarantees the graph can never loop forever (protecting both correctness and your API quota). Plus `main.py` catches exceptions and prints actionable tips instead of a stack trace.

## 💸 How this project minimizes API calls

- Supervisor only invokes agents the request actually needs (1 routing call ≈ 10 tokens of output).
- Retrieval is local (no embedding calls); only relevant note chunks go into prompts.
- SQLite response caching makes repeated test runs free.
- `MOCK_MODE=1` for fully offline development.
- A typical "quiz me" run = **2 LLM calls total** (1 route + 1 quiz + cached FINISH route).

## 📁 Repo structure

```
study-squad/
├── README.md
├── requirements.txt
├── .env.example          # copy to .env and add your free Groq key
├── main.py               # CLI entry point
├── state.py              # shared graph state (TypedDict)
├── llm.py                # LLM factory: Groq / mock mode / cache / retries
├── graph.py              # LangGraph wiring (Supervisor pattern)
├── agents/
│   ├── __init__.py
│   ├── supervisor.py     # routing agent + validation + loop cap
│   └── workers.py        # summarizer, quiz_master, planner agents
├── tools/
│   ├── __init__.py
│   └── retriever.py      # local zero-API mini-RAG retriever
└── sample_data/
    └── os_notes.txt      # sample Operating Systems class notes
```

## 🔮 Possible extensions
- Swap the local retriever for FAISS + embeddings for larger note sets.
- Add a Flashcard agent, or a Hierarchical layer (a "Subject Dean" supervising per-subject supervisors).
- Persist state with LangGraph checkpoints for multi-session studying.
