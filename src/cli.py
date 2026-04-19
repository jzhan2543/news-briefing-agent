"""
Command-line entry point for the briefing pipeline.

Usage:
    python -m src.cli "EU AI Act enforcement"

Prints the briefing markdown on success, or a structured failure message
on stderr with exit code 2 on failure. The thread_id is always printed so
the LangSmith trace is easy to find.
"""

import sys

from src.runner import run_briefing


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.cli <topic>", file=sys.stderr)
        sys.exit(1)

    topic = " ".join(sys.argv[1:])
    result = run_briefing(topic)

    if result.status == "success":
        print(result.briefing_markdown)
        print(f"\n---\nThread: {result.thread_id}")
    else:
        print(
            f"FAILURE ({result.reason}): {result.message}",
            file=sys.stderr,
        )
        if result.thread_id:
            print(f"Trace: {result.thread_id}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()