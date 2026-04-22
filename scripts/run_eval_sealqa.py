#!/usr/bin/env python3
"""Run full evaluation on SEAL-QA dataset."""
import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd

from src.agent_profiles import (
    Agent,
    sealqa_agent_options,
    make_sealqa_agent_options,
    set_sdk,
)
from src.cache import CacheConfig, RunCache
from src.evaluation.eval_full import evaluate_full, load_results
from src.evaluation.sealqa_scorer import score_sealqa
from src.schemas import AgentResponse
from src.skills import activate_skills_profile
from src.agent_profiles.skill_generator import get_project_root


async def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on SEAL-QA dataset")
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=Path(".dataset/seal-0.csv"),
        help="Path to SEAL-QA CSV file (default: .dataset/seal-0.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results/sealqa_eval_results.pkl"),
        help="Output pkl file path",
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=8,
        help="Max concurrent evaluations",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing results (start fresh)",
    )
    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        default="all",
        help="Filter by topic column ('all' or a specific topic value)",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=None,
        help="Limit to first N samples (default: all)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="claude-opus-4-5-20251101",
        help="Model for agent (default: claude-opus-4-5-20251101)",
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
    parser.add_argument(
        "--sdk",
        type=str,
        choices=["claude", "opencode", "openai"],
        default="claude",
        help="SDK to use: 'claude', 'opencode', or 'openai' (default: claude)",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="Override OPENAI_BASE_URL for --sdk openai (e.g., http://host:port/v1)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="Override OPENAI_API_KEY for --sdk openai",
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default="openai/gpt-5-mini",
        help="Grader model passed to dspy.LM (default: openai/gpt-5-mini)",
    )
    parser.add_argument(
        "--grader-base-url",
        type=str,
        default=None,
        help="Grader API base URL. Defaults to OPENAI_BASE_URL / --openai-base-url when unset.",
    )
    parser.add_argument(
        "--grader-api-key",
        type=str,
        default=None,
        help="Grader API key. Defaults to OPENAI_API_KEY / --openai-api-key when unset.",
    )
    parser.add_argument(
        "--skills-dir",
        type=str,
        default=".claude/skills_profiles/sealqa",
        help="Task-specific skills profile directory (default: .claude/skills_profiles/sealqa)",
    )
    parser.add_argument(
        "--debug-eval",
        action="store_true",
        help="Enable per-question evaluation heartbeat logs.",
    )
    parser.add_argument(
        "--eval-heartbeat-sec",
        type=int,
        default=30,
        help="Heartbeat interval (seconds) for --debug-eval logs (default: 30).",
    )
    args = parser.parse_args()

    if args.debug_eval:
        os.environ["EVOSKILL_EVAL_DEBUG"] = "1"
        os.environ["EVOSKILL_EVAL_HEARTBEAT_SEC"] = str(max(5, args.eval_heartbeat_sec))
        print(
            f"Eval debug enabled: heartbeat={os.environ['EVOSKILL_EVAL_HEARTBEAT_SEC']}s, "
            f"eval_timeout={os.getenv('EVOSKILL_EVAL_TIMEOUT_SEC', '1800')}s"
        )

    if args.sdk == "openai":
        if args.openai_base_url is not None:
            os.environ["OPENAI_BASE_URL"] = args.openai_base_url.strip()
        if args.openai_api_key is not None:
            os.environ["OPENAI_API_KEY"] = args.openai_api_key.strip()

    # Set SDK
    set_sdk(args.sdk)

    effective_openai_base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    effective_openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or None
    grader_base_url = (
        args.grader_base_url.strip()
        if args.grader_base_url is not None
        else effective_openai_base_url
    )
    grader_api_key = (
        args.grader_api_key.strip()
        if args.grader_api_key is not None
        else effective_openai_api_key
    )

    # Load dataset
    data = pd.read_csv(args.dataset)
    activate_skills_profile(get_project_root(), args.skills_dir)

    # Filter by topic if requested
    if args.topic != "all":
        data = data[data["topic"] == args.topic]

    # Limit to num_samples if specified
    if args.num_samples is not None:
        data = data.head(args.num_samples)

    print(f"Dataset: {len(data)} samples (topic={args.topic})")

    # Prepare items: (index, question, answer)
    # System prompt is loaded from prompt.txt via the agent options factory
    items = [
        (
            idx,
            row["question"],
            row["answer"],
        )
        for idx, row in data.iterrows()
    ]

    # Create agent and run
    agent_options_factory = make_sealqa_agent_options(model=args.model)
    agent = Agent(agent_options_factory, AgentResponse)

    model_info = f" (model: {args.model})" if args.model else " (model: opus)"
    print(f"Agent configured{model_info}")

    cache = None
    if not args.no_cache:
        cache = RunCache(CacheConfig(cache_dir=args.cache_dir))
        print(f"Cache enabled: {args.cache_dir}")
    else:
        print("Cache disabled")

    results = await evaluate_full(
        agent=agent,
        items=items,
        output_path=args.output,
        max_concurrent=args.max_concurrent,
        resume=not args.no_resume,
        cache=cache,
    )

    # Summary and scoring
    all_results = load_results(args.output)
    successful = [r for r in all_results if r.error is None]
    failed = [r for r in all_results if r.error is not None]

    # Score successful results
    correct = 0
    for r in successful:
        if r.trace and r.trace.output and r.trace.output.final_answer:
            score = score_sealqa(
                r.question,
                str(r.ground_truth),
                str(r.trace.output.final_answer),
                grader_model=args.grader_model,
                grader_base_url=grader_base_url,
                grader_api_key=grader_api_key,
            )
            if score > 0:
                correct += 1

    print(f"\n{'='*50}")
    print(f"Total completed: {len(all_results)}/{len(data)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed indices: {[r.index for r in failed]}")
    print(f"Accuracy: {correct}/{len(successful)} ({correct/len(successful)*100:.1f}%)" if successful else "Accuracy: N/A (no successful results)")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
