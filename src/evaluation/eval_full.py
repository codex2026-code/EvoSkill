import asyncio
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from tqdm.asyncio import tqdm_asyncio

from src.agent_profiles.base import Agent, AgentTrace

if TYPE_CHECKING:
    from src.cache import RunCache

T = TypeVar("T")


@dataclass
class IndexedEvalResult(Generic[T]):
    """Result of evaluating a single question with dataset index."""

    index: int  # Dataset row index
    question: str
    ground_truth: str
    trace: AgentTrace[T] | None
    error: str | None  # Error message if failed


def load_results(path: Path) -> list[IndexedEvalResult]:
    """Load results from pkl file."""
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def get_successful_indices(path: Path) -> set[int]:
    """Get set of indices that completed successfully (no error)."""
    results = load_results(path)
    return {
        r.index for r in results
        if r.error is None and (r.trace is None or not r.trace.is_error)
    }


async def evaluate_full(
    agent: Agent[T],
    items: list[tuple[int, str, str]],  # (index, question, ground_truth)
    output_path: Path,
    max_concurrent: int = 5,
    resume: bool = True,
    *,
    cache: "RunCache | None" = None,
) -> list[IndexedEvalResult[T]]:
    """
    Run agent on multiple questions in parallel, saving incrementally.

    Args:
        agent: The agent to evaluate
        items: List of (index, question, ground_truth) tuples
        output_path: Path to save pkl results
        max_concurrent: Max concurrent agent runs (default 5)
        resume: If True, skip already-processed indices

    Returns:
        List of IndexedEvalResult (only newly processed if resuming)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter out already successful indices if resuming (re-run failed ones)
    if resume:
        successful = get_successful_indices(output_path)
        items_to_run = [(i, q, gt) for i, q, gt in items if i not in successful]

        # Remove failed results from file so they can be replaced
        failed_indices = {i for i, _, _ in items_to_run}
        existing = load_results(output_path)
        kept_results = [r for r in existing if r.index not in failed_indices]
        if len(kept_results) < len(existing):
            # Some failed results were removed, save the cleaned file
            with open(output_path, "wb") as f:
                pickle.dump(kept_results, f)
            print(f"Removed {len(existing) - len(kept_results)} failed results for re-run")

        items = items_to_run
        if successful:
            print(f"Resuming: {len(successful)} successful, {len(items)} to run")

    if not items:
        print("All items already processed!")
        return []

    semaphore = asyncio.Semaphore(max_concurrent)
    lock = asyncio.Lock()
    eval_timeout_sec = int(os.getenv("EVOSKILL_EVAL_TIMEOUT_SEC", "1800"))
    eval_timeout_sec = max(60, eval_timeout_sec)
    eval_timeout_min = eval_timeout_sec // 60
    debug_eval = os.getenv("EVOSKILL_EVAL_DEBUG", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    heartbeat_sec = int(os.getenv("EVOSKILL_EVAL_HEARTBEAT_SEC", "30"))
    heartbeat_sec = max(5, heartbeat_sec)
    active: dict[int, float] = {}
    completed = 0

    if debug_eval:
        print(
            f"[EVAL-DEBUG] items={len(items)} max_concurrent={max_concurrent} "
            f"timeout={eval_timeout_sec}s heartbeat={heartbeat_sec}s"
        )

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(heartbeat_sec)
            async with lock:
                if not active:
                    continue
                now = asyncio.get_event_loop().time()
                longest = sorted(
                    ((idx, now - started) for idx, started in active.items()),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
                longest_text = ", ".join(
                    f"idx={idx} elapsed={int(elapsed)}s" for idx, elapsed in longest
                )
                print(
                    f"[EVAL-DEBUG][HEARTBEAT] active={len(active)} completed={completed}/{len(items)} | "
                    f"longest: {longest_text}"
                )

    heartbeat_task: asyncio.Task[None] | None = None
    if debug_eval:
        heartbeat_task = asyncio.create_task(heartbeat())

    async def run_one(
        index: int, question: str, ground_truth: str
    ) -> IndexedEvalResult[T]:
        async with semaphore:
            error = None
            trace = None
            started = asyncio.get_event_loop().time()
            async with lock:
                active[index] = started
            if debug_eval:
                print(f"[EVAL-DEBUG][START] idx={index} q={question[:80]!r}")
            try:
                async with asyncio.timeout(eval_timeout_sec):
                    if cache is not None:
                        trace = cache.get(question, agent.response_model)

                    if trace is None:
                        trace = await agent.run(question)
                        if cache is not None:
                            cache.set(question, trace)
            except asyncio.TimeoutError:
                error = f"TimeoutError: Eval timed out after {eval_timeout_min} minutes"
                print(f"[TIMEOUT] Index {index}: {question[:50]}...")
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"
                print(f"[ERROR] Index {index}: {question[:50]}... -> {error}")

            result = IndexedEvalResult(
                index=index,
                question=question,
                ground_truth=ground_truth,
                trace=trace,
                error=error,
            )

            # Append to file immediately (thread-safe)
            async with lock:
                nonlocal completed
                completed += 1
                active.pop(index, None)
                existing = load_results(output_path)
                existing.append(result)
                with open(output_path, "wb") as f:
                    pickle.dump(existing, f)
                if debug_eval:
                    elapsed = asyncio.get_event_loop().time() - started
                    status = "ERROR" if error else "OK"
                    print(
                        f"[EVAL-DEBUG][DONE] idx={index} status={status} "
                        f"elapsed={elapsed:.1f}s completed={completed}/{len(items)}"
                    )

            return result

    tasks = [run_one(idx, q, gt) for idx, q, gt in items]
    try:
        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
    return results
