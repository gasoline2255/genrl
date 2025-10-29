from genrl.state import GameState
from dataclasses import dataclass
from typing import Any, List
import json
import re
import ollama

@dataclass
class RewardsOllamaConfig:
    model: str = "qwen2.5-coder:1.5b-instruct"
    temperature: float = 0.0
    num_predict: int = 256

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
    def __init__(self, ollama_config: RewardsOllamaConfig = RewardsOllamaConfig()):
        self.stage = 0
        self.model = ollama_config.model
        self.temperature = ollama_config.temperature
        self.num_predict = ollama_config.num_predict


    def _build_prompt(self, solution_code: str, unit_tests: str) -> str:
        return (
            "You are a code judge. You will be given Python code and unit tests. "
            "Decide if the unit tests will pass when run against the code. "
            "Respond ONLY with a JSON fenced block and nothing else. The JSON must have keys 'is_correct' (boolean) and 'reason' (string).\n\n"
            "Here is the candidate solution code:\n\n"
            f"```python\n{solution_code}\n```\n\n"
            "Here are the unit tests:\n\n"
            f"```python\n{unit_tests}\n```\n\n"
            "Now respond with ONLY the following format, no extra commentary:\n\n"
            "```json\n{\n  \"is_correct\": true | false,\n  \"reason\": \"brief justification\"\n}\n```"
        )

    def _extract_json(self, text: str) -> Any:
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)

    def reward_fn(self, solutions, unittests):
        """Compute rewards for solutions by executing them against unit tests.
        
        Args:
            solutions: List of code solutions to test.
            unittests: Corresponding unit tests to run.
            
        Returns:
            List of reward values (1.0 for success, 0.0 for failure).
        """
        rewards = []
        for solution in solutions:
            if solution == 'No python fence found in solution':
                rewards.append(0.0)
                continue
            try:
                prompt = self._build_prompt(str(solution), str(unittests))
                response = ollama.generate(model=self.model, prompt=prompt, options={"temperature": self.temperature, "num_predict": self.num_predict})
                raw_text = response.response
                data = self._extract_json(raw_text)
                is_correct = bool(data.get("is_correct", False))
                rewards.append(1.0 if is_correct else 0.0)
            except Exception:
                rewards.append(0.0)

        return rewards



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
