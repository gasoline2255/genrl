
from dataclasses import dataclass
import copy
import json
import logging
import os
import shutil
import re
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_sandbox import SyncPyodideSandbox

logger = logging.getLogger(__name__)


try:
    from vllm import LLM as VLLMEngine
    from vllm import SamplingParams as VLLMSamplingParams
    _VLLM_AVAILABLE = True
except Exception:
    VLLMEngine = None
    VLLMSamplingParams = None
    _VLLM_AVAILABLE = False

SYSTEM_PROMPT = "You are an expert software engineer and team leader who is responsible for interviewing junior devs.  You must provide the coding task that the junior dev will be asked to solve."
PROMPT_TEXT = (
            "You are to propose a Python coding question and matching unit tests.\n"
            "Constraints:\n"
            "- Do NOT provide any solution or implementation.\n"
            "- Choose a single clear function name and write a one-sentence description of what it should do.\n"
            "- Provide runnable unit tests only (pytest or unittest) that validate expected behavior of the function.\n"
            "- The tests should be self-contained and reference the function name but not implement it.\n"
            "- The output MUST be valid JSON with exactly these keys: question (string) and tests (string).\n"
            "- The tests field must contain ONLY Python code.\n"
            "- Do not include any additional keys or commentary.\n"
            "- The question should contain the function name that the tests refer to.\n"
            "- IMPORTANT: Wrap your entire output in a Markdown fenced code block that starts with ```json on its own line and ends with ``` on its own line.\n"
            "- Do not include any text outside the fenced block.\n\n"
            "Example output format (fenced JSON):\n"
            "```json\n{\n  \"question\": \"Write a function foo(x) that ...\",\n  \"tests\": \"import pytest\\n\\n def test_foo(): ...\"\n}\n```"
            "The following are problems you have already proposed, please do not repeat:\n\n"
        )


def parse_json_from_fence(text):
    """
    Parses a JSON fence block from a given text string.
    Args:
        text (str): The input text containing a JSON fence block.
    Returns:
        dict or list or None: The parsed JSON object, or None if no valid JSON block is found.
    """
    # Regex to find a block starting with ```json and ending with ```
    # The `?` makes the match non-greedy, so it stops at the first closing fence.
    # The `re.DOTALL` flag allows the `.` to match newlines.
    match = re.search(r'```json(.*?)```', text, re.DOTALL)
    if match:
        json_string = match.group(1).strip()
        try:
            # Use json.loads to parse the cleaned string
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError as e:
            logger.info(f"Error decoding JSON: {e}")
            return None
    return None

@dataclass
class PPOConfig:
    clip_range: float = 0.2
    ppo_epochs: int = 2
    learning_rate: float = 1e-5

@dataclass
class VllmConfig:
    use_vllm: bool = False
    vllm_engine: VLLMEngine = None
    vllm_sampling: VLLMSamplingParams = VLLMSamplingParams(max_tokens=1024) if _VLLM_AVAILABLE else None
    # vLLM resource controls (tunable to mitigate OOM)
    vllm_gpu_memory_utilization: float = 0.4  # fraction of available GPU memory vLLM may use
    vllm_max_model_len: int = 4096          # reduce KV cache footprint if needed
    vllm_swap_space: int = 4                # GB of CPU swap for KV cache spillover
    vllm_tensor_parallel_size: int = 1      # keep to 1 unless you shard across GPUs


class Proposer:
    def __init__(self, model_path, ppo_config: PPOConfig = PPOConfig(), vllm_config: VllmConfig = VllmConfig()):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        self.model.to(self.device, dtype=self.train_dtype)
        self.sandbox = SyncPyodideSandbox(allow_net=True)

        # PPO config and optimizer
        self.ppo_config = ppo_config
        self.optimizer = AdamW(self.model.parameters(), lr=self.ppo_config.learning_rate)

        # vLLM related attributes
        self.vllm_config = vllm_config
        self._vllm_available = _VLLM_AVAILABLE and self.vllm_config.use_vllm
        logger.info(f"Using Vllm for inference: {self._vllm_available}")
        self._vllm_engine = None

        # Track current model path (HF id or local dir after training)
        self._current_model_path = model_path

        self.prompts = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEXT},
        ]

        self.previous_problems = []

    def _ensure_vllm_engine(self):
        """Create vLLM engine if available and not already initialized."""
        if not self._vllm_available:
            return
        if self._vllm_engine is None:
            # Instantiate vLLM engine with the current model path
            self._vllm_engine = VLLMEngine(
                model=self._current_model_path,
                dtype='auto',
                gpu_memory_utilization=self.vllm_config.vllm_gpu_memory_utilization,
                max_model_len=self.vllm_config.vllm_max_model_len,
                swap_space=self.vllm_config.vllm_swap_space,
                tensor_parallel_size=self.vllm_config.vllm_tensor_parallel_size,
            )

    def _shutdown_vllm_engine(self):
        """Tear down vLLM engine to free GPU memory before training."""
        if self._vllm_engine is not None:
            try:
                # Best-effort teardown; vLLM frees on GC
                del self._vllm_engine
            finally:
                self._vllm_engine = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def checkpoint_model(self, save_dir: str = "./proposer_ckpt"):
        # Save to disk
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        self._current_model_path = save_dir

    def _reload_vllm_engine_from_hf(self):
        """
        Save HF model/tokenizer to a directory and recreate vLLM engine from it.
        """
        # Recreate engine
        if not self._vllm_available:
            return
        self._shutdown_vllm_engine()
        self._ensure_vllm_engine()

    def generate_proposal(self):

        # Build a single prompt string using the tokenizer's chat template


        proposal = None
        proposal_raw = None
        # Prefer vLLM if available

        prompt = copy.deepcopy(self.prompts)
        prompt[1]['content'] += '\n'.join(self.previous_problems) + '\n'
        if self._vllm_available:
            self._ensure_vllm_engine()
            prompt_text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors=None,
            )
            while proposal is None:
                outputs = self._vllm_engine.generate([prompt_text], self.vllm_config.vllm_sampling)
                text = outputs[0].outputs[0].text
                proposal_raw = text
                proposal = parse_json_from_fence(proposal_raw)
                # TODO: Validate the proposal
                if proposal:
                    logger.info(f'testing proposal {proposal}')
                    validation = self.sandbox.execute(proposal["tests"]) # should run through interpreter without errors
                    if validation.stderr or validation.status == 'error':
                        proposal = None
        else:
            # Fallback to HF generate if vLLM not available
            input_ids = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            prompt_length = input_ids.size(1)
            input_ids = input_ids.to(self.model.device)
            while proposal is None:
                response = self.model.generate(
                    input_ids,
                    max_new_tokens=1024,
                )
                generated_ids = response[0][prompt_length:]
                proposal_raw = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True,
                )
                proposal = parse_json_from_fence(proposal_raw)

                # TODO: Validate the proposal
                if proposal:
                    logger.info(f'testing proposal {proposal}')
                    validation = self.sandbox.execute(proposal["tests"]) # should run through interpreter without errors
                    if validation.stderr or validation.status == 'error':
                        proposal = None

        self.previous_problems.append(proposal['question'])
        
        proposal["proposal_raw"] = proposal_raw
        return proposal

    def reward_fn(self, solver_rewards: list[float]) -> float:
        if len(solver_rewards) == 0 or solver_rewards is None:
            return None
        elif len(solver_rewards) == 1:
            return solver_rewards[0]
        else:
            avg_reward = sum(solver_rewards) / len(solver_rewards)
            if avg_reward in [0.0, 1.0]:
                return 0.0
            else:
                return 1 - avg_reward

    def _logprob_sum_for_generated(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        gen_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute log-probabilities for the generated token sequence under the model.
        input_ids: [B, T_in] or [1, T_in]
        generated_ids: [B, T_gen] or [1, T_gen]
        gen_attention_mask: optional [B, T_gen] (1 for real tokens, 0 for pad)
        Returns: [B, T_gen, V] log-probs for each generated token position.
        """
        if generated_ids.numel() == 0:
            # Return an empty logprob tensor with correct batch dims
            batch = input_ids.shape[0]
            return torch.zeros((batch, 0, self.model.config.vocab_size), device=self.device, dtype=self.train_dtype)

        batch = input_ids.shape[0]

        prompt_attention = torch.ones_like(input_ids, device=input_ids.device)
        if gen_attention_mask is None:
            gen_attention_mask = torch.ones_like(generated_ids, device=input_ids.device)

        context = torch.cat([input_ids, generated_ids], dim=1)  # [B, T_in + T_gen]
        attention_mask = torch.cat([prompt_attention, gen_attention_mask], dim=1)  # [B, T_in + T_gen]
        outputs = self.model(context, attention_mask=attention_mask)
        logits = outputs.logits[:, -generated_ids.shape[1]:, :]  # [B, T_gen, V]
        logprobs = F.log_softmax(logits, dim=-1)
        return logprobs  # [B, T_gen, V]

    def train(self, solver_rewards, proposal=None):
        """
        Perform a PPO update using the proposed trajectory and the provided reward(s).
        - Supports two modes:
          1) Single sample: solver_rewards: list[float], proposal: str
          2) Batched: solver_rewards: list[list[float]], proposal: list[str]
        Padding and masking are applied for batched proposals.
        """
        if proposal is None:
            logger.info("No proposal provided to train on.")
            return
        is_batched = isinstance(proposal, list)
        logger.info(f'is batched: {is_batched}, proposals: {len(proposal)}')
        # Before training, shut down vLLM to free memory; we'll reload after.
        self._shutdown_vllm_engine()

        # Tokenize the chat prompt and the generated completion(s)
        self.model.eval()
        with torch.no_grad():
            base_input_ids = self.tokenizer.apply_chat_template(
                self.prompts,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)

            if not is_batched:
                # Single-sample path (backwards compatible)
                gen_batch = self.tokenizer(
                    proposal,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                gen_ids = gen_batch["input_ids"].to(self.device)  # [1, T_gen]
                input_ids = base_input_ids  # [1, T_in]
            else:
                # Batched path
                gen_batch = self.tokenizer(
                    proposal,
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding=True,
                )
                gen_ids = gen_batch["input_ids"].to(self.device)  # [B, T_gen]
                batch_size = gen_ids.size(0)
                input_ids = base_input_ids.repeat(batch_size, 1)  # [B, T_in]

            gen_attn = gen_batch.get("attention_mask", torch.ones_like(gen_ids)).to(self.device)
            old_logprob = self._logprob_sum_for_generated(input_ids, gen_ids, gen_attention_mask=gen_attn).detach().to(self.device)
            selected_old_logprobs = old_logprob.gather(dim=-1, index=gen_ids.unsqueeze(-1)).squeeze(-1)  # [B | 1, T_gen]

        # Compute advantages
        if not is_batched:
            reward_scalar = self.reward_fn(solver_rewards)
            if reward_scalar is None:
                # Nothing to learn from this sample
                return
            advantage = torch.tensor([reward_scalar], dtype=torch.float32, device=self.device)  # [1]
        else:
            # solver_rewards: list[list[float]] -> compute per-sample scalar
            adv_list = []
            for rlist in solver_rewards:
                rs = self.reward_fn(rlist)
                adv_list.append(0.0 if rs is None else rs)
            advantage = torch.tensor(adv_list, dtype=torch.float32, device=self.device)  # [B]
        
        logger.info(f'advantage: {advantage}')
        mask = gen_attn.float()  # [B | 1, T_gen]

        valid_counts = mask.sum(dim=1).clamp_min(1.0)  # [B | 1]

        self.model.train()
        for _ in range(self.ppo_config.ppo_epochs):
            self.optimizer.zero_grad()
            # Recompute current logprob under updated policy
            new_logprob = self._logprob_sum_for_generated(input_ids, gen_ids, gen_attention_mask=gen_attn)
            selected_new_logprobs = new_logprob.gather(dim=-1, index=gen_ids.unsqueeze(-1)).squeeze(-1)  # [B | 1, T_gen]

            # PPO ratio per token
            ratio = torch.exp(selected_new_logprobs - selected_old_logprobs)  # [B | 1, T_gen]

            # Clipped objective per token, scale by per-sample advantage
            adv_expanded = advantage.unsqueeze(-1)  # [B | 1, 1]
            unclipped = ratio * adv_expanded  # [B | 1, T_gen]
            clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_config.clip_range, 1.0 + self.ppo_config.clip_range)
            clipped = clipped_ratio * adv_expanded  # [B | 1, T_gen]
            per_token_loss = -torch.min(unclipped, clipped)  # [B | 1, T_gen]

            # Mask padding and compute mean per sequence, then mean over batch
            masked_loss = (per_token_loss * mask).sum(dim=1) / valid_counts  # [B]
            policy_loss = masked_loss.mean()  # scalar
            logger.info(f'policy_loss: {policy_loss.item()}, advantage: {advantage}')
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.checkpoint_model()
        # After training, reload vLLM engine from updated HF weights
        self._reload_vllm_engine_from_hf()


if __name__ == "__main__":
    proposer = Proposer("Qwen/Qwen3-0.6B")
    proposer.generate_proposal()
   