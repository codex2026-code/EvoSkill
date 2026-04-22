import asyncio
import os
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from tqdm.asyncio import tqdm_asyncio

from src.agent_profiles.base import Agent, AgentTrace

if TYPE_CHECKING:
    from src.cache import RunCache

T = TypeVar("T")


@dataclass
class EvalResult(Generic[T]):
    """Result of evaluating a single question."""
    question: str
    ground_truth: str
    trace: AgentTrace[T] | None


async def evaluate_agent_parallel(
    agent: Agent[T],
    items: list[tuple[str, str]],
    max_concurrent: int = 2,
    *,
    cache: "RunCache | None" = None,
) -> list[EvalResult[T]]:
    """
    Run agent on multiple questions in parallel.

    Args:
        agent: The agent to evaluate
        items: List of (question, ground_truth) tuples
        max_concurrent: Max concurrent agent runs (default 2)
        cache: Optional RunCache for caching results (keys on git tree hash)

    Returns:
        List of EvalResult containing question, ground_truth, and trace
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    debug_eval = os.getenv("EVOSKILL_EVAL_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    heartbeat_enabled = os.getenv("EVOSKILL_EVAL_HEARTBEAT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    heartbeat_sec = int(os.getenv("EVOSKILL_EVAL_HEARTBEAT_SEC", "30"))
    heartbeat_sec = max(5, heartbeat_sec)
    eval_timeout_sec = int(os.getenv("EVOSKILL_EVAL_TIMEOUT_SEC", "1800"))
    eval_timeout_sec = max(60, eval_timeout_sec)
    eval_timeout_min = eval_timeout_sec // 60
    if debug_eval or heartbeat_enabled:
        print(
            "[EVAL][CONFIG] "
            f"items={len(items)} max_concurrent={max_concurrent} "
            f"agent_timeout_sec={getattr(agent, 'timeout_seconds', 'unknown')} "
            f"eval_timeout_sec={eval_timeout_sec} heartbeat_sec={heartbeat_sec}"
        )

    async def run_one(question: str, ground_truth: str) -> EvalResult[T]:
        async with semaphore:
            started_at = time.monotonic()
            if debug_eval:
                print(f"[EVAL][START] q='{question[:80]}'")

            heartbeat_task: asyncio.Task[None] | None = None

            async def _heartbeat() -> None:
                while True:
                    await asyncio.sleep(heartbeat_sec)
                    elapsed = time.monotonic() - started_at
                    print(f"[EVAL][WAIT] q='{question[:60]}' elapsed={elapsed:.1f}s")

            try:
                async with asyncio.timeout(eval_timeout_sec):
                    # Check cache first
                    trace = None
                    if cache is not None:
                        trace = cache.get(question, agent.response_model)
                        if debug_eval and trace is not None:
                            print(f"[EVAL][CACHE_HIT] q='{question[:80]}'")

                    # Cache miss - run agent
                    if trace is None:
                        if debug_eval or heartbeat_enabled:
                            heartbeat_task = asyncio.create_task(_heartbeat())
                        if debug_eval:
                            print(f"[EVAL][RUN] q='{question[:80]}' invoking agent.run")
                        trace = await agent.run(question)
                        if debug_eval:
                            print(f"[EVAL][RUN_DONE] q='{question[:80]}'")
                        # Store in cache
                        if cache is not None:
                            cache.set(question, trace)

            except asyncio.TimeoutError:
                print(f"Eval timed out ({eval_timeout_min}min) for: {question[:50]}...")
                trace = None
            except Exception as e:
                print(f"Failed on question: {question[:50]}... Error: {e}")
                trace = None
            finally:
                if heartbeat_task is not None:
                    heartbeat_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await heartbeat_task
                if debug_eval:
                    elapsed = time.monotonic() - started_at
                    outcome = "ok" if trace is not None else "none"
                    print(f"[EVAL][END] q='{question[:80]}' elapsed={elapsed:.1f}s trace={outcome}")
            return EvalResult(question=question, ground_truth=ground_truth, trace=trace)

    tasks = [run_one(q, gt) for q, gt in items]
    results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
    return results
