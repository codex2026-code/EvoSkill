#!/usr/bin/env python3
"""Run self-improving agent loop."""

import asyncio

import pandas as pd

from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.agent_profiles import (
    Agent,
    base_agent_options,
    proposer_options,
    skill_generator_options,
    prompt_generator_options,
)
from src.agent_profiles.skill_generator import get_project_root
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    ProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)


async def main():
    data = pd.read_csv('.dataset/train_set.csv')

    train = data.sample(20, random_state=42)
    val = data.drop(train.index).sample(5, random_state=31)

    train_data = [(row.question, row.ground_truth) for _, row in train.iterrows()]
    val_data = [(row.question, row.ground_truth) for _, row in val.iterrows()]

    agents = LoopAgents(
        base=Agent(base_agent_options, AgentResponse),
        proposer=Agent(proposer_options, ProposerResponse),
        skill_generator=Agent(skill_generator_options, ToolGeneratorResponse),
        prompt_generator=Agent(prompt_generator_options, PromptGeneratorResponse),
    )
    manager = ProgramManager(cwd=get_project_root())

    config = LoopConfig(
        max_iterations=20,
        frontier_size=3,
        no_improvement_limit=5,
        concurrency=4,
        cache_enabled=True,
        reset_feedback=True,
    )

    loop = SelfImprovingLoop(config, agents, manager, train_data, val_data)
    result = await loop.run()

    print(f"Best: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")


if __name__ == "__main__":
    asyncio.run(main())
