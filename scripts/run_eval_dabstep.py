#!/usr/bin/env python3
"""Run full evaluation on Dabstep dataset."""
import argparse
import asyncio
import hashlib
import json
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
from src.skills import activate_skills_profile
from src.agent_profiles.skill_generator import get_project_root


PROMPT = """You are an expert data analyst and you will answer factoid questions by loading and referencing the files/documents listed below.
You have these files available:
{context_files}

Here is the question you need to answer:
{question}

Here are the guidelines you must follow when answering the question above:
{guidelines}
"""


def is_empty_answer(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "na", "n/a", "withheld"}


def summarize_answer_column(data: pd.DataFrame) -> dict[str, float | int | str | bool]:
    answers = data["answer"] if "answer" in data.columns else pd.Series([None] * len(data))
    nonempty_mask = ~answers.map(is_empty_answer)
    nonempty_count = int(nonempty_mask.sum())
    nonempty_values = answers[nonempty_mask].astype(str).str.strip()
    unique_nonempty = int(nonempty_values.nunique()) if nonempty_count else 0
    nonempty_ratio = (nonempty_count / len(data)) if len(data) else 0.0
    dominant_ratio = 0.0
    dominant_value = ""
    if nonempty_count:
        counts = nonempty_values.value_counts(dropna=False)
        dominant_value = str(counts.index[0])
        dominant_ratio = float(counts.iloc[0] / nonempty_count)
    return {
        "row_count": int(len(data)),
        "answer_nonempty_count": nonempty_count,
        "answer_unique_count": unique_nonempty,
        "answer_nonempty_ratio": float(nonempty_ratio),
        "answer_dominant_value": dominant_value,
        "answer_dominant_ratio": float(dominant_ratio),
        "looks_withheld": (nonempty_count == 0) or (unique_nonempty <= 1) or (nonempty_ratio < 0.2),
    }


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata_path_for_output(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".meta.json")


def validate_resume_metadata(metadata_path: Path, current_metadata: dict, resume: bool) -> None:
    if not resume:
        return
    if not metadata_path.exists():
        return
    old = json.loads(metadata_path.read_text(encoding="utf-8"))
    keys = ["dataset_path", "dataset_sha256", "requested_split", "row_count"]
    mismatched = [k for k in keys if old.get(k) != current_metadata.get(k)]
    if mismatched:
        raise ValueError(
            "Refusing resume: dataset metadata changed for "
            f"{', '.join(mismatched)}. Remove old pkl/meta or run with --no-resume."
        )


def write_predictions_jsonl(results, output_path: Path) -> Path:
    pred_path = output_path.with_suffix(output_path.suffix + ".predictions.jsonl")
    with pred_path.open("w", encoding="utf-8") as f:
        for r in results:
            final_answer = None
            if r.trace and r.trace.output:
                final_answer = getattr(r.trace.output, "final_answer", None)
            rec = {
                "task_id": r.index,
                "final_answer": final_answer,
                "error": r.error,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return pred_path


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
    parser.add_argument(
        "--skills-dir",
        type=str,
        default=".claude/skills_profiles/dabstep",
        help="Task-specific skills profile directory (default: .claude/skills_profiles/dabstep)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="default",
        choices=["default", "dev", "all"],
        help="Requested split to evaluate (default/dev/all).",
    )
    parser.add_argument(
        "--allow-default-local-scoring",
        action="store_true",
        help="Allow local scoring on default split only when answers are present and valid.",
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
    raw_data = pd.read_csv(args.dataset)
    data = raw_data
    activate_skills_profile(get_project_root(), args.skills_dir)

    split_table = None
    if "split" in raw_data.columns:
        print("\nSplit summary (from CSV):")
        print("split\trow_count\tanswer_nonempty_count\tanswer_unique_count")
        split_table = []
        for split_name, part in raw_data.groupby(raw_data["split"].astype(str).str.lower(), dropna=False):
            stats = summarize_answer_column(part)
            split_table.append((str(split_name), stats))
            print(
                f"{split_name}\t{stats['row_count']}\t{stats['answer_nonempty_count']}\t{stats['answer_unique_count']}"
            )

    if "split" in data.columns and args.split != "all":
        data = data[data["split"].astype(str).str.lower() == args.split]

    if args.split == "default" and len(data) == 460:
        raise ValueError(
            "Detected 460 rows for split=default. 你可能传入了 tasks 全量而不是 default split。"
        )
    if args.split == "dev" and len(data) != 10:
        raise ValueError(
            f"split=dev expects 10 rows, got {len(data)}. 请检查是否传入了错误 CSV/split。"
        )

    answer_stats = summarize_answer_column(data)

    should_local_score = args.split != "default" or args.allow_default_local_scoring
    if args.split == "default" and should_local_score and answer_stats["looks_withheld"]:
        raise ValueError(
            "DABStep default split is for submission-style evaluation; use dev for local scoring or provide an answered CSV."
        )

    # Filter by level if requested
    if args.level != "all":
        data = data[data["level"].astype(str) == args.level]

    # Limit to num_samples if specified
    if args.num_samples is not None:
        data = data.head(args.num_samples)

    print(f"Dataset: {len(data)} samples (level={args.level})")

    run_metadata = {
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": source_sha256(args.dataset),
        "requested_split": args.split,
        "row_count": int(len(data)),
        "answer_stats": answer_stats,
        "split_table": [
            {
                "split": split_name,
                **stats,
            }
            for split_name, stats in (split_table or [])
        ],
    }
    meta_path = metadata_path_for_output(args.output)
    validate_resume_metadata(meta_path, run_metadata, resume=not args.no_resume)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

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
    unscored = [
        r
        for r in successful
        if not (r.trace and r.trace.output and getattr(r.trace.output, "final_answer", None))
    ]

    # Score successful results
    correct = 0
    scored = 0
    for r in successful:
        if r.trace and r.trace.output and r.trace.output.final_answer:
            scored += 1
            score = question_scorer(str(r.trace.output.final_answer), str(r.ground_truth))
            if score:
                correct += 1

    prediction_path = write_predictions_jsonl(all_results, args.output)

    print(f"\n{'='*50}")
    print(f"Total completed: {len(all_results)}/{len(data)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Unscored (missing final_answer): {len(unscored)}")
    if failed:
        print(f"Failed task_ids: {[r.index for r in failed]}")
    if unscored:
        print("Examples with empty final_answer:")
        for r in unscored[:20]:
            snippet = str(r.question).replace("\n", " ")[:120]
            print(f"  - task_id={r.index} | question={snippet}")

    if should_local_score:
        print(
            f"Accuracy: {correct}/{scored} ({(correct/scored*100):.1f}%)"
            if scored
            else "Accuracy: N/A (no scored items with final_answer)"
        )
    else:
        print(
            "Accuracy: SKIPPED (default split submission mode). "
            "Use --split dev for local scoring or provide answered CSV + --allow-default-local-scoring."
        )

    print(f"Results saved to: {args.output}")
    print(f"Predictions saved to: {prediction_path}")
    print(f"Run metadata saved to: {meta_path}")


if __name__ == "__main__":
    asyncio.run(main())
