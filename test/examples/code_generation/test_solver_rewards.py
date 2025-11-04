import pytest
from genrl.examples.code_generation.solver_rewards import CodeGenerationRewards


@pytest.fixture()
def rewards_with_model():
    solver_tokenizer_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    solver_token_lim = 512
    r = CodeGenerationRewards(solver_tokenizer_path, solver_token_lim)
    return r


def test_build_prompt_contains_sections(rewards_with_model):
    question = "Write a python function that takes two numbers as input and returns their addition."
    code = "def add(a, b):\n    return a + b"
    tests = """import unittest
class T(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1,2), 3)
"""
    dataset = 'mbpp'
    prompt = rewards_with_model._build_prompt(dataset, code, tests, question)
    assert "You are an expert programming evaluator" in prompt
    assert "```json" in prompt
    assert code in prompt
    assert tests in prompt


def test_reward_fn_handles_no_python_fence_sentinel(rewards_with_model):
    solutions = ["No python fence found in solution"]
    unit_tests = "irrelevant"
    question = "irrelevant"
    dataset = 'test dataset'
    rewards = rewards_with_model.reward_fn(dataset, solutions, unit_tests, question)
    assert rewards == [-0.8]


def test_reward_fn_empty_solutions_returns_empty_list(rewards_with_model):
    solutions = []
    unit_tests = "does not matter"
    question = "irrelevant"
    dataset = 'test dataset'
    rewards = rewards_with_model.reward_fn(dataset, solutions, unit_tests, question)
    assert rewards == []


def test_reward_fn_multiple_all_sentinel(rewards_with_model):
    solutions = [
        "No python fence found in solution",
        "No python fence found in solution",
    ]
    unit_tests = "irrelevant"
    question = "irrelevant"
    dataset = 'test dataset'
    rewards = rewards_with_model.reward_fn(dataset, solutions, unit_tests, question)
    assert rewards == [-0.8, -0.8]
