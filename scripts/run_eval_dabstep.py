#!/usr/bin/env python3
"""Run full evaluation on Dabstep dataset."""
import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd

from src.agent_profiles import (
    Agent,
    dabstep_agent_options,
    make_dabstep_agent_options,
    set_sdk,
)
from src.cache import CacheConfig, RunCache
from src.evaluation.eval_full import evaluate_full, load_results
from src.evaluation.dabstep_scorer import question_scorer
from src.schemas import AgentResponse


PROMPT = """You are an expert data analyst and you will answer factoid questions by loading and referencing the files/documents listed below.
You have these files available:
{context_files}

Here is the question you need to answer:
{question}

Here are the guidelines you must follow when answering the question above:
{guidelines}
"""


async def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on Dabstep dataset")
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=Path(".dataset/dabstep_data.csv"),
        help="Path to dabstep CSV file (default: .dataset/dabstep_data.csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="DABstep-data/data/context",
        help="Path to shared context files directory (default: DABstep-data/data/context)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results/dabstep_eval_results.pkl"),
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
        "--level",
        "-l",
        type=str,
        default="all",
        help="Filter by level column ('all' or a specific level value)",
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
    args = parser.parse_args()

    # Set SDK
    set_sdk(args.sdk)
    if args.sdk == "openai":
        if args.model:
            os.environ["OPENAI_MODEL"] = args.model.strip()
        if args.openai_base_url is not None:
            os.environ["OPENAI_BASE_URL"] = args.openai_base_url.strip()
        if args.openai_api_key is not None:
            os.environ["OPENAI_API_KEY"] = args.openai_api_key.strip()

        effective_base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
        if effective_base_url:
            print(f"OpenAI endpoint: {effective_base_url}")
        else:
            print("OpenAI endpoint: https://api.openai.com/v1 (default)")
        print(f"OpenAI model: {(os.getenv('OPENAI_MODEL') or '').strip() or '<unset>'}")

    # Load dataset
    data = pd.read_csv(args.dataset)

    # Filter by level if requested
    if args.level != "all":
        data = data[data["level"].astype(str) == args.level]

    # Limit to num_samples if specified
    if args.num_samples is not None:
        data = data.head(args.num_samples)

    print(f"Dataset: {len(data)} samples (level={args.level})")

    # Auto-discover context files from data-dir
    data_dir = Path(args.data_dir).resolve()
    context_file_names = sorted(f.name for f in data_dir.iterdir() if f.is_file())
    context_files_text = "\n".join(f"- {data_dir / name}" for name in context_file_names)
    print(f"Context files ({len(context_file_names)}): {', '.join(context_file_names)}")

    # Prepare items: (task_id, formatted_prompt, answer)
    items = [
        (
            row["task_id"],
            PROMPT.format(
                context_files=context_files_text,
                question=row["question"],
                guidelines=row["guidelines"],
            ),
            row["answer"],
        )
        for _, row in data.iterrows()
    ]

    # Create agent and run
    agent_options_factory = make_dabstep_agent_options(model=args.model, data_dir=args.data_dir)
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
            score = question_scorer(str(r.trace.output.final_answer), str(r.ground_truth))
            if score:
                correct += 1

    print(f"\n{'='*50}")
    print(f"Total completed: {len(all_results)}/{len(data)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed task_ids: {[r.index for r in failed]}")
    print(f"Accuracy: {correct}/{len(successful)} ({correct/len(successful)*100:.1f}%)" if successful else "Accuracy: N/A (no successful results)")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
