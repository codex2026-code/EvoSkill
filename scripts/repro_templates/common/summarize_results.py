#!/usr/bin/env python3
"""Summarize eval pkl outputs into JSON/CSV for reproducible comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.evaluation.eval_full import load_results
from src.evaluation.dabstep_scorer import question_scorer
from src.evaluation.sealqa_scorer import score_sealqa
from src.evaluation.livecodebench import score_livecodebench


def _score(task: str, question: str, ground_truth: str, predicted: str, args: argparse.Namespace) -> float:
    if task == "dabstep":
        return 1.0 if question_scorer(predicted, ground_truth) else 0.0
    if task == "livecodebench":
        return score_livecodebench(question, ground_truth, predicted)
    if task == "sealqa":
        return score_sealqa(
            question,
            ground_truth,
            predicted,
            grader_model=args.grader_model,
            grader_base_url=args.grader_base_url,
            grader_api_key=args.grader_api_key,
        )
    raise ValueError(f"Unsupported task: {task}")


def summarize(task: str, result_pkl: Path, args: argparse.Namespace) -> dict[str, Any]:
    all_results = load_results(result_pkl)
    successful = [r for r in all_results if r.error is None]
    failed = [r for r in all_results if r.error is not None]

    correct = 0
    scored = 0
    for r in successful:
        if r.trace and r.trace.output and r.trace.output.final_answer:
            scored += 1
            pred = str(r.trace.output.final_answer)
            score = _score(task, str(r.question), str(r.ground_truth), pred, args)
            if score > 0:
                correct += 1

    accuracy = (correct / scored) if scored else 0.0
    return {
        "task": task,
        "result_pkl": str(result_pkl),
        "total": len(all_results),
        "successful": len(successful),
        "failed": len(failed),
        "scored": scored,
        "correct": correct,
        "accuracy": accuracy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize eval result pkl files")
    parser.add_argument("--task", choices=["sealqa", "dabstep", "livecodebench"], required=True)
    parser.add_argument("--result-pkl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--append-csv", type=Path, default=None)

    # For sealqa parity with run_eval_sealqa.py scorer options.
    parser.add_argument("--grader-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--grader-base-url", type=str, default=None)
    parser.add_argument("--grader-api-key", type=str, default=None)
    return parser.parse_args()


def maybe_append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    columns = [
        "task",
        "result_pkl",
        "total",
        "successful",
        "failed",
        "scored",
        "correct",
        "accuracy",
    ]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row[k] for k in columns})


def main() -> None:
    args = parse_args()
    summary = summarize(args.task, args.result_pkl, args)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.append_csv is not None:
        maybe_append_csv(args.append_csv, summary)
        print(f"Appended summary row to: {args.append_csv}")


if __name__ == "__main__":
    main()
