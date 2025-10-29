import pytest
from genrl.examples.code_generation.solver_rewards import CodeGenerationRewards
import ollama


@pytest.fixture()
def rewards_with_model():
    r = CodeGenerationRewards()
    return r


def test_extract_json_from_fenced_response(rewards_with_model):
    text = """```json
{
  "is_correct": true,
  "reason": "All tests should pass."
}
```"""
    data = rewards_with_model._extract_json(text)
    assert isinstance(data, dict)
    assert data["is_correct"] is True
    assert "reason" in data


def test_build_prompt_contains_sections(rewards_with_model):
    code = "def add(a, b):\n    return a + b"
    tests = """import unittest
class T(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1,2), 3)
"""
    prompt = rewards_with_model._build_prompt(code, tests)
    assert "You are a code judge" in prompt
    assert "```python" in prompt
    assert code in prompt
    assert tests in prompt


def test_reward_fn_handles_no_python_fence_sentinel(rewards_with_model):
    solutions = ["No python fence found in solution"]
    unit_tests = "irrelevant"
    rewards = rewards_with_model.reward_fn(solutions, unit_tests)
    assert rewards == [0.0]


def test_reward_fn_empty_solutions_returns_empty_list(rewards_with_model):
    solutions = []
    unit_tests = "does not matter"
    rewards = rewards_with_model.reward_fn(solutions, unit_tests)
    assert rewards == []


def test_reward_fn_multiple_all_sentinel(rewards_with_model):
    solutions = [
        "No python fence found in solution",
        "No python fence found in solution",
    ]
    unit_tests = "irrelevant"
    rewards = rewards_with_model.reward_fn(solutions, unit_tests)
    assert rewards == [0.0, 0.0]


@pytest.mark.integration
def test_reward_fn_positive_single_when_ollama_available():
    # Skip if Ollama isn't available
    try:
        _ = ollama.list()
    except Exception:
        pytest.skip("Ollama server not available; skipping positive integration test.")

    r = CodeGenerationRewards()
    solutions = [
        """def add(a, b):
    return a + b
""",
    ]
    unit_tests = """import unittest
class T(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1,2), 3)
"""

    rewards = r.reward_fn(solutions, unit_tests)
    assert rewards == [1.0]


@pytest.mark.integration
def test_reward_fn_positive_multiple_when_ollama_available():
    # Skip if Ollama isn't available
    try:
        _ = ollama.list()
    except Exception:
        pytest.skip("Ollama server not available; skipping positive integration test.")

    r = CodeGenerationRewards()
    solutions = [
        """def add(a, b):
    return a + b
""",
        """def mul(a, b):
    return a / b
""",
    ]
    unit_tests = """import unittest
class T(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2,3), 5)
    def test_mul(self):
        self.assertEqual(mul(2,4), 8)
"""

    rewards = r.reward_fn(solutions, unit_tests)
    assert rewards == [1.0, 0.0]
