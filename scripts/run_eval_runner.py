#!/usr/bin/env python3
"""Run EvalRunner from CLI with selectable SDK (claude/opencode/openai).

This is a thin wrapper around `src.api.EvalRunner` so users can run the
high-level async API in the same configuration style as existing eval scripts.

Example (OpenAI-compatible local endpoint):
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY= \
python scripts/run_eval_runner.py \
  --sdk openai \
  --task sealqa \
  --model deepseek-v3.2 \
  --dataset .dataset/seal-0.csv \
  --output results/sealqa_eval_local_openai.pkl \
  --max-concurrent 1 \
  --num-samples 2
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.agent_profiles import set_sdk
from src.api import EvalRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run high-level EvalRunner with explicit SDK/model/task settings",
    )
    parser.add_argument(
        "--sdk",
        choices=["claude", "opencode", "openai"],
        default="claude",
        help="SDK backend to use (default: claude)",
    )
    parser.add_argument(
        "--task",
        default="sealqa",
        help="Registered task name (default: sealqa)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset CSV path (default: use task default dataset)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for the selected SDK (e.g. gpt-4.1, deepseek-v3.2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/eval_runner_results.pkl"),
        help="Output .pkl path (default: results/eval_runner_results.pkl)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Max concurrent runs (default: 8)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Run only first N rows (default: all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode (default: resume enabled)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache lookups/writes under .cache/runs",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/runs"),
        help="Cache directory (default: .cache/runs)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    set_sdk(args.sdk)

    runner = EvalRunner(
        dataset=str(args.dataset) if args.dataset else None,
        task=args.task,
        model=args.model,
        output=str(args.output),
        max_concurrent=args.max_concurrent,
        resume=not args.no_resume,
        num_samples=args.num_samples,
        cache_enabled=not args.no_cache,
        cache_dir=str(args.cache_dir),
    )

    summary = await runner.run()
    print(
        f"Final Accuracy: {summary.accuracy:.1%} "
        f"({summary.correct}/{summary.successful})"
    )


if __name__ == "__main__":
    asyncio.run(main())
