#!/usr/bin/env python3
"""
tools/test_fit_step1.py — Test harness for the fit Step 1 spatial reasoning agent.

Loads a plan JSON and calls the agent with the plan and its own boundary as the
target (self-fit test). Prints the proposed room positions.

Usage:
    python tools/test_fit_step1.py
    python tools/test_fit_step1.py --plan data/fixtures/plans/test_data/290.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.floorplan_core.fit_step1 import run_fit_step1_agent

BRIEF_PATH = ROOT / "docs" / "dev" / "design" / "fit-step1-brief.md"
DEFAULT_PLAN = ROOT / "data" / "fixtures" / "plans" / "test_data" / "987.json"


def run_test(plan_path: Path, dry_run: bool = False):
    if not dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "Error: ANTHROPIC_API_KEY is not set.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    plan = json.loads(plan_path.read_text())
    brief = BRIEF_PATH.read_text()

    # Use the plan's own boundary as the target for a self-fit test
    target_boundary = [[v[0], v[1]] for v in plan["boundary"]]

    if not dry_run:
        print(f"Plan     : {plan_path.name}")
        print(f"Rooms    : {len(plan['rooms'])}")
        print(f"Edges    : {len(plan.get('edges', []))}")
        print(f"Boundary : {len(target_boundary)} vertices")
        print()

    result = run_fit_step1_agent(
        plan=plan,
        target_boundary=target_boundary,
        brief_text=brief,
        dry_run=dry_run,
    )

    positions = result["positions"]
    print()
    print("Proposed positions")
    print("-" * 48)
    room_by_id = {r["id"]: r.get("label", r["id"]) for r in plan["rooms"]}
    for room_id, pos in sorted(positions.items()):
        print(f"  {room_id:8s}  ({pos['x']:.4f}, {pos['y']:.4f})  {room_by_id.get(room_id, room_id)}")

    missing = [r["id"] for r in plan["rooms"] if r["id"] not in positions]
    if missing:
        print(f"\nWARNING: no position proposed for: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description="Fit Step 1 — agent test")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print system prompt and user message then exit")
    args = parser.parse_args()
    run_test(args.plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
