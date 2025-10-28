from genrl.state import GameState
from typing import Any, List
from langchain_sandbox import PyodideSandbox

from genrl.logging_utils.global_defs import get_logger
import asyncio
import re

def get_solutions(
    game_state: GameState, stage: int
) -> dict[Any, dict[Any, List[Any]]]:
    actions = game_state.get_stage_actions(stage)
    solutions = {}
    for agent in actions:
        solutions[agent] = {}  
        for batch_id in actions[agent]:
            solutions[agent][batch_id] = []
            for node, _ in enumerate(actions[agent][batch_id]):
                solutions[agent][batch_id].append(actions[agent][batch_id][node])
    return solutions  # Indices are [Agent][Batch Item][Node Idx][Solution]


def get_unittests(game_state: GameState, stage: int) -> dict[Any, dict[Any, List[Any]]]:
    world_states = game_state.get_stage_state(stage)
    unittests = {}  # Key per agent
    for agent in world_states:
        unittests[agent] = {} 
        for batch_id in world_states[agent]:
            unittests[agent][batch_id] = []
            for node, _ in enumerate(world_states[agent][batch_id]):
                unittests[agent][batch_id].append(
                    world_states[agent][batch_id][node].environment_states["answer"]
                )
    return unittests  # Indices are [Agent][Batch Item][Node Idx]


class CodeGenerationRewards:
    def __init__(self):
        self.stage = 0
        self.sandbox = PyodideSandbox(allow_net=True)

    def reward_fn(self, solutions, unittests):
        # Sandboxed environment executes unit tests with solution
        rewards = []
        for solution in solutions:
            if solution == 'No python fence found in solution':
                rewards.append(0.0)
                continue


            # result = self.sandbox.execute(str(solution) + "\n\n" + str(unittests))
            code = str(solution) + "\n\n" + str(unittests)
            result = asyncio.run(self.run_sandbox(code=code, timeout=30))
            if result is None:
                rewards.append(0.0)
            else:
                errors = result.stderr
                status = result.status
                if errors or status == 'error':
                    rewards.append(0.0)
                else:
                    rewards.append(1.0)

        return rewards
    

    async def run_sandbox(self, code: str, timeout: int):
        try:
            result = await asyncio.wait_for(self.sandbox.execute(code), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            get_logger().info(f"Code timed out after {timeout} seconds.")
            return None



    def __call__(self, game_state):
        solutions_by_agent = get_solutions(game_state, self.stage)
        unittests_by_agent = get_unittests(game_state, self.stage)
        rewards = {}  # Key per agent
        for agent in solutions_by_agent:
            rewards[agent] = {}  # Will store a list per batch item
            for batch_id in solutions_by_agent[agent]:
                rewards[agent][batch_id] = []
                for node_idx, _ in enumerate(solutions_by_agent[agent][batch_id]):
                    rewards[agent][batch_id].append(
                        self.reward_fn(
                            solutions_by_agent[agent][batch_id][node_idx],
                            unittests_by_agent[agent][batch_id][node_idx],
                        )
                    )
        return rewards
