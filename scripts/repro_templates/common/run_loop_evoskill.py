#!/usr/bin/env python3
"""Run EvoSkill loop for tasks without dedicated loop CLI scripts.

Official logic alignment:
- Uses src.api.EvoSkill (task registry driven).
- Uses EVOSKILL_SKILLS_DIR to persist profile-specific skills.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from src.api import EvoSkill
from src.agent_profiles import set_sdk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EvoSkill loop via API (dabstep/livecodebench template)"
    )
    parser.add_argument(
        "--task",
        choices=["dabstep", "livecodebench"],
        required=True,
        help="Task name registered in src.api.task_registry",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset path override. If omitted, task default_dataset is used.",
    )
    parser.add_argument(
        "--skills-dir",
        type=str,
        required=True,
        help="Skills profile directory for this run (mapped via EVOSKILL_SKILLS_DIR).",
    )
    parser.add_argument("--mode", choices=["skill_only", "prompt_only"], default="skill_only")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--frontier-size", type=int, default=3)
    parser.add_argument("--no-improvement-limit", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--failure-samples", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.18)
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--continue-mode", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-reset-feedback", action="store_true")
    parser.add_argument(
        "--selection-strategy",
        choices=["best", "random", "round_robin"],
        default="best",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--sdk",
        choices=["claude", "opencode", "openai"],
        default="claude",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="Override OPENAI_BASE_URL for --sdk openai (e.g., http://host:port/v1).",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="Override OPENAI_API_KEY for --sdk openai.",
    )
    parser.add_argument(
        "--loop-summary-json",
        type=Path,
        required=True,
        help="Path for serialized loop summary JSON.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Keep skills profile isolated per experiment branch.
    os.environ["EVOSKILL_SKILLS_DIR"] = args.skills_dir

    if args.sdk == "openai":
        if args.model:
            os.environ["OPENAI_MODEL"] = args.model.strip()
        if args.openai_base_url is not None:
            os.environ["OPENAI_BASE_URL"] = args.openai_base_url.strip()
        if args.openai_api_key is not None:
            os.environ["OPENAI_API_KEY"] = args.openai_api_key.strip()

    # Follow official SDK switch path.
    set_sdk(args.sdk)

    loop = EvoSkill(
        dataset=args.dataset,
        task=args.task,
        model=args.model,
        mode=args.mode,
        max_iterations=args.max_iterations,
        frontier_size=args.frontier_size,
        no_improvement_limit=args.no_improvement_limit,
        concurrency=args.concurrency,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        continue_mode=args.continue_mode,
        cache_enabled=not args.no_cache,
        reset_feedback=not args.no_reset_feedback,
        failure_samples=args.failure_samples,
        selection_strategy=args.selection_strategy,
    )
    result = await loop.run()

    payload = {
        "task": args.task,
        "dataset": args.dataset,
        "skills_dir": args.skills_dir,
        "mode": args.mode,
        "model": args.model,
        "sdk": args.sdk,
        "openai_base_url": args.openai_base_url,
        "max_iterations": args.max_iterations,
        "frontier_size": args.frontier_size,
        "no_improvement_limit": args.no_improvement_limit,
        "concurrency": args.concurrency,
        "failure_samples": args.failure_samples,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "continue_mode": args.continue_mode,
        "cache_enabled": not args.no_cache,
        "reset_feedback": not args.no_reset_feedback,
        "selection_strategy": args.selection_strategy,
        "best_program": result.best_program,
        "best_score": result.best_score,
        "frontier": result.frontier,
        "iterations_completed": result.iterations_completed,
        "history": result.history,
    }

    args.loop_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.loop_summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Loop summary saved to: {args.loop_summary_json}")


if __name__ == "__main__":
    asyncio.run(main())
