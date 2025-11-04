from genrl.state import GameState
from dataclasses import dataclass
from typing import Any, List
import ollama
from transformers import AutoTokenizer
from genrl.examples.code_generation.solver_utils import check_eos, parse_python_fence, build_prompt, parse_response, get_solutions, get_unittests, get_questions

@dataclass
class RewardsOllamaConfig:
    model: str = "qwen2.5-coder:1.5b-instruct"
    temperature: float = 0.0
    num_predict: int = 512
    
class CodeGenerationRewards:
    def __init__(self, solver_tokenizer_path: str, solver_token_lim: int, ollama_config: RewardsOllamaConfig = RewardsOllamaConfig()):
        self.stage = 0
        self.model = ollama_config.model
        self.temperature = ollama_config.temperature
        self.num_predict = ollama_config.num_predict
        self.tokenizer = AutoTokenizer.from_pretrained(solver_tokenizer_path, padding_side="left")
        self.solver_token_lim = solver_token_lim


    def reward_fn(self, solutions, unit_tests, question):
        """Compute rewards for solutions by executing them against unit tests.
        
        Args:
            solutions: List of code solutions to test.
            unittests: Corresponding unit tests to run.
            
        Returns:
            List of reward values (1.0 for success, 0.0 for failure).
        """
        rewards = []
        for solution in solutions:
            if not isinstance(solution, str):
                reward = -1.2
            else:
                parsed_code = parse_python_fence(solution)
                eos_found = check_eos(solution, self.tokenizer, self.solver_token_lim)
                if parsed_code is None: # No fenced code found
                    reward = -1.0
                else:
                    prompt = build_prompt(question, solution, unit_tests)
                    response = ollama.generate(model=self.model, prompt=prompt, options={"temperature": self.temperature, "num_predict": self.num_predict})
                    raw_text = response.response
                    try:
                        reward = parse_response(raw_text)
                        if reward is None:
                            reward = 0.0
                    except:
                        reward = 0.0
                reward += 0.2 if eos_found else -0.2
            rewards.append(reward)

        return rewards



    def __call__(self, game_state):
        solutions_by_agent = get_solutions(game_state, self.stage)
        unittests_by_agent = get_unittests(game_state, self.stage)
        questions = get_questions(game_state, self.stage)

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
                            questions[agent][batch_id][node_idx]
                        )
                    )
        return rewards
