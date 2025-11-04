import pytest
from genrl.examples.code_generation.solver_rewards import CodeGenerationRewards
from genrl.examples.code_generation.solver_utils import build_prompt
import ollama
import yaml


@pytest.fixture()
def rewards_with_model():
    solver_tokenizer_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    solver_token_lim = 512
    r = CodeGenerationRewards(solver_tokenizer_path, solver_token_lim)
    return r


def test_build_prompt_contains_sections(rewards_with_model):
    question = "Write a pyhton function that takes two numbers as input and returns their addition."
    code = "def add(a, b):\n    return a + b"
    tests = """import unittest
class T(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1,2), 3)
"""
    prompt = build_prompt(question, code, tests)
    assert "You are an expert programming evaluator" in prompt
    assert "```json" in prompt
    assert question in prompt
    assert code in prompt
    assert tests in prompt


def test_reward_fn_handles_no_python_fence_sentinel(rewards_with_model):
    solutions = ["No python fence found in solution"]
    unit_tests = "irrelevant"
    question = "irrelevant"
    rewards = rewards_with_model.reward_fn(solutions, unit_tests, question)
    assert rewards == [-0.8]


def test_reward_fn_empty_solutions_returns_empty_list(rewards_with_model):
    solutions = []
    unit_tests = "does not matter"
    question = "irrelevant"
    rewards = rewards_with_model.reward_fn(solutions, unit_tests, question)
    assert rewards == []


def test_reward_fn_multiple_all_sentinel(rewards_with_model):
    solutions = [
        "No python fence found in solution",
        "No python fence found in solution",
    ]
    unit_tests = "irrelevant"
    question = "irrelevant"
    rewards = rewards_with_model.reward_fn(solutions, unit_tests, question)
    assert rewards == [-0.8, -0.8]


@pytest.mark.integration
def test_reward_fn_positive_single_when_ollama_available():
    # Skip if Ollama isn't available
    try:
        _ = ollama.list()
    except Exception:
        pytest.skip("Ollama server not available; skipping positive integration test.")

    solver_tokenizer_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    solver_token_lim = 512
    r = CodeGenerationRewards(solver_tokenizer_path, solver_token_lim)

    solutions = ["```python def add(a, b):\n   return a + b```"]
    unit_tests = "assert add(2,3) == 5"
    question = "Write a Python function named `add` that takes two arguments and returns their sum."

    rewards = r.reward_fn(solutions, unit_tests, question)
    assert rewards == [1.2]


@pytest.mark.integration
def test_reward_fn_positive_multiple_when_ollama_available():
    # Skip if Ollama isn't available
    try:
        _ = ollama.list()
    except Exception:
        pytest.skip("Ollama server not available; skipping positive integration test.")

    solver_tokenizer_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    solver_token_lim = 512
    r = CodeGenerationRewards(solver_tokenizer_path, solver_token_lim)

    solutions = [
        "```python def add(a, b):\n   return a + b```",
        "```python def mul(a, b):\n   return a/b```",
    ]
    unit_tests = "assert add(2,3) == 5"
    question = "Write a Python function named `add` that takes two arguments and returns their sum."

    rewards = r.reward_fn(solutions, unit_tests, question)
    assert rewards == [1.2, 0.2]
