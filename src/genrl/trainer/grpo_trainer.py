import gc
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from genrl.trainer.trainer_utils import DTYPE_MAP


def create_reference_model(model: torch.nn.Module) -> torch.nn.Module:
    ref_model = deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model.eval()


@dataclass
class GRPOTrainerConfig:
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    beta: float = 0.0
    temperature: float = 1.0
    dtype: str = "float32"
    enable_gradient_checkpointing: bool = True
    max_new_tokens: int = 256
    num_generations: int = 2
    learning_rate: float = 1e-6
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    ppo_epochs: int = 1
    minibatch_size: int = 2


class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
    """
    GRPO trainer with vLLM support for inference (generation) to avoid CUDA OOM.
    """

    def __init__(self, models: List[Any], config: GRPOTrainerConfig, **kwargs):
        if not models or len(models) < 1:
            raise ValueError("At least one model must be provided")

        self.args = config
        self.use_vllm = kwargs.get("use_vllm", True)   # ENABLE vLLM BY DEFAULT

        self.callbacks = kwargs.get("callbacks", [])
        self.save_dir = kwargs.get("log_dir", "./outputs")
        self.global_step = 0
        self.dtype = DTYPE_MAP[self.args.dtype]

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # ---- VLLM INITIALIZATION ----
        if self.use_vllm:
            try:
                from vllm import LLM, SamplingParams
            except ImportError:
                raise ImportError("vLLM not installed. Install it with: pip install vllm")

            model_name = models[0] if isinstance(models[0], str) else models[0].config._name_or_path
            self.vllm_engine = LLM(
                model=model_name,
                dtype="float16",
                gpu_memory_utilization=0.90,
            )

            self.vllm_sampling_params = SamplingParams(
                n=self.args.num_generations,
                max_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
            )

            self.processing_class = self.vllm_engine.get_tokenizer()
            self.model = None
            self.ref_model = None
            self.optimizer = None

            print("🔥 Using vLLM for generation (NO CUDA OOM)")
        else:
            # ---- STANDARD HUGGINGFACE TRAINING ----
            self.model = models[0]
            self.model = self.model.to(device=self.device, dtype=self.dtype)

            if self.args.enable_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()

            self.ref_model = None
            if self.args.beta != 0.0:
                self.ref_model = create_reference_model(self.model).to(
                    device=self.device, dtype=self.dtype
                )

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

            self.processing_class = AutoTokenizer.from_pretrained(
                self.model.config._name_or_path, padding_side="left"
            )

        # Common components
        self._initialize_metrics()
        self._initialize_generation_config()
        self.init_tracker(self.save_dir, log_with=kwargs.get("log_with", None))

    # ------------------------------------------------------
    # GENERATION WITH vLLM OR HUGGINGFACE
    # ------------------------------------------------------
    def generate(self, inputs: Any, return_completion_ids: bool = False, stage=0):
        """Use vLLM for generation to avoid CUDA OOM."""
        prompts = []

        for item in inputs:
            text = item["prompt"]
            prompt = self.processing_class.apply_chat_template(
                text, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        # ---- VLLM GENERATE ----
        if self.use_vllm:
            outputs = self.vllm_engine.generate(prompts, self.vllm_sampling_params)
            rollout = [[o.text for o in out.outputs] for out in outputs]

            if return_completion_ids:
                rollout_ids = [[o.token_ids for o in out.outputs] for out in outputs]
                return rollout, rollout_ids

            return rollout

        # ---- FALLBACK HF GENERATE ----
        input_tokens = self.processing_class(
            text=prompts, return_tensors="pt", padding=True
        )
        input_tokens = input_tokens.to(self.device)

        rollout = []
        rollout_ids = []
        for _ in range(self.args.num_generations):
            outputs = self.model.generate(
                input_ids=input_tokens.input_ids,
                attention_mask=input_tokens.attention_mask,
                generation_config=self.generation_config,
            )

            prompt_length = input_tokens.input_ids.size(1)
            comp_ids = outputs[:, prompt_length:]
            completions = self.processing_class.batch_decode(
                comp_ids, skip_special_tokens=True
            )

            if not rollout:
                rollout = [[c] for c in completions]
                rollout_ids = [[comp_ids[i]] for_]()]()_
