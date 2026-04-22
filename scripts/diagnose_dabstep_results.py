#!/usr/bin/env python3
"""Diagnose DABStep evaluation PKL outputs and dataset CSV integrity."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_CLASS_CACHE: dict[tuple[str, str], type] = {}


def _placeholder_class(module: str, name: str) -> type:
    key = (module, name)
    if key not in _CLASS_CACHE:
        _CLASS_CACHE[key] = type(name, (), {})
    return _CLASS_CACHE[key]


class PermissiveUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins":
            return super().find_class(module, name)
        try:
            return super().find_class(module, name)
        except Exception:
            return _placeholder_class(module, name)


def load_results(path: Path) -> list[Any]:
    with path.open("rb") as f:
        return PermissiveUnpickler(f).load()


def is_empty_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "withheld", "n/a", "na"}


def get_final_answer(record: Any) -> Any:
    trace = getattr(record, "trace", None)
    output = getattr(trace, "output", None) if trace is not None else None
    return getattr(output, "final_answer", None) if output is not None else None


def summarize(records: list[Any], label: str) -> dict[str, Any]:
    successful = [r for r in records if getattr(r, "error", None) is None]
    failed = [r for r in records if getattr(r, "error", None) is not None]
    scored = [r for r in successful if not is_empty_like(get_final_answer(r))]

    print(f"\n=== {label} records ===")
    print("task_id\tground_truth\tfinal_answer\terror\thas_final_answer")
    for r in records:
        task_id = getattr(r, "index", None)
        gt = getattr(r, "ground_truth", None)
        pred = get_final_answer(r)
        err = getattr(r, "error", None)
        has_pred = not is_empty_like(pred)
        print(f"{task_id}\t{repr(gt)}\t{repr(pred)}\t{repr(err)}\t{has_pred}")

    print(
        f"Summary ({label}): total={len(records)}, successful={len(successful)}, failed={len(failed)}, scored={len(scored)}"
    )

    return {
        "total": len(records),
        "successful": len(successful),
        "failed": len(failed),
        "scored": len(scored),
        "task_ids": {getattr(r, "index", None) for r in records},
    }


def inspect_ground_truth(full_records: list[Any]) -> None:
    gts = [getattr(r, "ground_truth", None) for r in full_records]
    empty_count = sum(is_empty_like(g) for g in gts)
    normalized = [str(g).strip().lower() if not is_empty_like(g) else "<EMPTY>" for g in gts]
    counts = Counter(normalized)

    print("\n=== Full ground_truth diagnostics ===")
    print(f"empty_like={empty_count}/{len(gts)} ({(empty_count/len(gts))*100:.2f}%)")
    print("top_values:")
    for v, c in counts.most_common(10):
        print(f"  {v!r}: {c}")


def compare_overlap(full_records: list[Any], dev_records: list[Any]) -> None:
    full_by_id = {getattr(r, "index", None): r for r in full_records}
    dev_by_id = {getattr(r, "index", None): r for r in dev_records}

    overlap_ids = sorted(set(full_by_id) & set(dev_by_id))
    print("\n=== full/dev overlap ===")
    print(f"overlap_count={len(overlap_ids)}")
    print(f"overlap_task_ids={overlap_ids}")

    gt_mismatch = []
    for task_id in overlap_ids:
        full_gt = getattr(full_by_id[task_id], "ground_truth", None)
        dev_gt = getattr(dev_by_id[task_id], "ground_truth", None)
        if str(full_gt) != str(dev_gt):
            gt_mismatch.append((task_id, full_gt, dev_gt))

    if gt_mismatch:
        print("ground_truth mismatches on overlap:")
        for task_id, full_gt, dev_gt in gt_mismatch:
            print(f"  task_id={task_id}: full={full_gt!r}, dev={dev_gt!r}")
    else:
        print("ground_truth一致: overlap task_ids 上 full/dev 完全一致")

    print(f"full contains all dev tasks: {set(dev_by_id).issubset(set(full_by_id))}")


def dataset_split_table(dataset_csv: Path) -> None:
    with dataset_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("\n[dataset] 空文件")
        return
    if "split" not in rows[0]:
        print("\n[dataset] 无 split 列，无法按 split 输出 row_count/answer 统计")
        return

    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        split = str(row.get("split", "")).strip().lower()
        by_split[split].append(row)

    print("\n=== Dataset split table ===")
    print("split\trow_count\tanswer_nonempty_count\tanswer_unique_count")
    for split, part in sorted(by_split.items(), key=lambda x: x[0]):
        answers = [row.get("answer", "") for row in part]
        nonempty = [a for a in answers if not is_empty_like(a)]
        print(f"{split}\t{len(part)}\t{len(nonempty)}\t{len(set(str(a).strip() for a in nonempty))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, default=Path("results.pkl"))
    parser.add_argument("--dev", type=Path, default=Path("results-dev.pkl"))
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()

    full_records = load_results(args.full)
    dev_records = load_results(args.dev)

    summarize(full_records, "full")
    summarize(dev_records, "dev")
    inspect_ground_truth(full_records)
    compare_overlap(full_records, dev_records)

    if args.dataset is not None:
        dataset_split_table(args.dataset)


if __name__ == "__main__":
    main()
