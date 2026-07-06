"""StudySquad — multi-agent exam prep assistant (Supervisor pattern).

Usage:
    python main.py --notes sample_data/os_notes.txt --request "summarize and make a quiz"
    python main.py --request "make me a 3 day study plan for operating systems"
"""

import argparse
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(description="StudySquad multi-agent study helper")
    parser.add_argument("--notes", default="sample_data/os_notes.txt",
                        help="Path to a .txt file with your class notes")
    parser.add_argument("--request", required=True,
                        help='What you want, e.g. "summarize my notes and quiz me"')
    args = parser.parse_args()

    notes_path = pathlib.Path(args.notes)
    if not notes_path.exists():
        print(f"[warn] notes file '{args.notes}' not found — continuing without notes")
        notes_text = ""
    else:
        notes_text = notes_path.read_text(encoding="utf-8")

    # import here so a missing API key gives a clean message, not a stack trace
    try:
        from graph import build_graph
        app = build_graph()
        result = app.invoke(
            {
                "user_request": args.request,
                "notes_text": notes_text,
                "completed": [],
                "steps_taken": 0,
            }
        )
    except Exception as e:
        print(f"\n[error] Run failed: {e}")
        print("Tips: check GROQ_API_KEY in .env, or set MOCK_MODE=1 to test without a key.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STUDYSQUAD RESULT")
    print("=" * 60 + "\n")
    print(result["final_answer"])
    print("\n[agents that ran]:", ", ".join(result.get("completed", [])) or "none")


if __name__ == "__main__":
    main()
