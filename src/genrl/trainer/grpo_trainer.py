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

    def __init__(self, models: List[Any], config: GRPOTrainerConfig, **kwargs):

        if not models or len(models) < 1:
            raise ValueError("At least one model must be provided")

        self.model = models[0]
        self.args = config

        # ============================
        # 🔥 vLLM INFERENCE SUPPORT
        # ============================
        self.use_vllm = kwargs.get("use_vllm", False)
        self.vllm_engine = None
        self.vllm_sampling = None

        if self.use_vllm:
            print(">> Initializing vLLM inference engine...")
            try:
                from vllm import LLM, SamplingParams
            except ImportError:
                raise ImportError("vLLM not installed. Run: pip install vllm")

            self.vllm_engine = LLL = LLM(
                model=self.model.config._name_or_path,
                dtype="float16",
                gpu_memory_utilization=kwargs.get("vllm_gpu_memory", 0.85),
            )
            self.vllm_sampling = SamplingParams(
                n=self.args.num_generations,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                max_tokens=self.args.max_new_tokens,
            )
            print(">> vLLM active for generate() ONLY")

        # ======================================
        # OPTIMIZER (HF training stays unchanged)
        # ======================================
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

        self.processing_class = kwargs.get("processing_class", None)
        self.callbacks = kwargs.get("callbacks", [])
        self.save_dir = kwargs.get("log_dir", "./outputs")
        self.global_step = 0
        self.dtype = DTYPE_MAP[self.args.dtype]
        self.enable_gradient_checkpointing = self.args.enable_gradient_checkpointing

        # ================
        # DEVICE SELECTION
        # ================
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # ============================
        # MODEL & TOKENIZER SETUP
        # ============================
        self._initialize_model(self.enable_gradient_checkpointing)
        self._initialize_tokenizers()
        self._initialize_metrics()
        self._initialize_generation_config()

        self.init_tracker(self.save_dir, log_with=kwargs.get("log_with", None))

    # ----------------------------------------------------
    # Model / tokenizer setup
    # ----------------------------------------------------
    def _initialize_model(self, enable_gradient_checkpointing):
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        if enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = create_reference_model(self.model).to(
                device=self.device, dtype=self.dtype
            )

    def _initialize_tokenizers(self):
        if self.processing_class is None:
            self.processing_class = AutoTokenizer.from_pretrained(
                self.model.config._name_or_path, padding_side="left"
            )

    def _initialize_metrics(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

    def _initialize_generation_config(self):
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            bos_token_id=self.processing_class.bos_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            min_p=self.args.min_p,
            repetition_penalty=self.args.repetition_penalty,
        )

    # ----------------------------------------------------
    # Input formatting
    # ----------------------------------------------------
    def _process_inputs(self, inputs, with_template=True, for_training=False):

        # Unify inputs
        if hasattr(inputs, "to_dict"):
            inputs = [dict(inputs[i]) for i in range(len(inputs))]
        elif isinstance(inputs, dict):
            inputs = [inputs]

        # Build templated prompts
        if with_template:
            if for_training:
                templated_prompts = []
                for item in inputs:
                    for _ in range(self.args.num_generations):
                        templated_prompts.append(
                            self.processing_class.apply_chat_template(
                                item["prompt"], tokenize=False, add_generation_prompt=True
                            )
                        )
            else:
                templated_prompts = [
                    self.processing_class.apply_chat_template(
                        item["prompt"], tokenize=False, add_generation_prompt=True
                    )
                    for item in inputs
                ]

        # ---- vLLM MODE: return raw strings, NOT tensors ----
        if self.use_vllm and not for_training:
            return templated_prompts

        # HF tokenization (training OR no vLLM)
        return self.processing_class(
            text=templated_prompts, return_tensors="pt", padding=True, truncation=True
        )

    # ----------------------------------------------------
    # GENERATION (vLLM or HF)
    # ----------------------------------------------------
    def generate(self, inputs: Any, return_completion_ids: bool = False, stage=0):

        # ==========================================================
        # 🔥 vLLM inference path — prevents CUDA OOM!
        # ==========================================================
        if self.use_vllm:
            prompts = self._process_inputs(inputs)  # returns list[str]

            outputs = self.vllm_engine.generate(prompts, self.vllm_sampling)

            rollout = [[o.text for o in out.outputs] for out in outputs]
            return rollout  # no token IDs needed for GRPO

        # ==========================================================
        # HF fallback for training mode (logprobs needed)
        # ==========================================================
        input_tokens = self._process_inputs(inputs)
        rollout, rollout_ids = [], []

        for _ in range(self.args.num_generations):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_tokens.input_ids.to(self.model.device),
                    attention_mask=input_tokens.attention_mask.to(self.model.device),
                    generation_config=self.generation_config,
                )

            prompt_len = input_tokens.input_ids.size(1)
            completion_ids = outputs[:, prompt_len:]
            completions = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )

            if len(rollout) == 0:
                rollout = [[comp] for comp in completions]
            else:
                for idx, comp in enumerate(completions):
                    rollout[idx].append(comp)

        return rollout

    # ----------------------------------------------------
    # GRPO LOGPROB CALCULATION
    # ----------------------------------------------------
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits

        logits = logits[:, :-1, :]  # drop final logits
        labels = input_ids[:, -logits_to_keep:].contiguous()

        logits = logits[:, -logits_to_keep:].contiguous()
        loss_mask = attention_mask[:, -logits_to_keep:].to(logits.dtype)

        token_log_probs = -torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
        ).view(logits.size(0), logits.size(1))

        token_log_probs = token_log_probs * loss_mask
        return token_log_probs

    # ----------------------------------------------------
    # GRPO LOSS
    # ----------------------------------------------------
    def compute_loss(self, model, inputs, mode="train", return_metrics=False):

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(
            self.model.device
        )

        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        advantages = inputs["advantages"]
        old_per_token_logps = per_token_logps.detach()

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1, 1 - self.args.epsilon, 1 + self.args.epsilon_high
        )

        advantages = advantages.unsqueeze(dim=-1)
        per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
        per_token_loss = per_token_loss * completion_mask

        loss = per_token_loss.sum() / completion_mask.sum()

        self._metrics[mode]["loss"].append(loss.item())

        if return_metrics:
            return loss, {"loss": loss.item()}
        return loss

    # ----------------------------------------------------
    # TRAINING LOOP
    # ----------------------------------------------------
    def train(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ) -> None:

        self.model.train()
        global_step = self.global_step

        for stage in range(state.stage):
            global_step = self.step(stage, state, data_manager, reward_manager, global_step)

        self.global_step = global_step
        self.model.eval()

    def step(
        self,
        stage: int,
        state: GameState,
        data_manager: DataManager,
        reward_manager: RewardManager,
        global_step: int,
    ) -> int:

        global_step += 1

        stage_inputs = state.get_stage_state(stage)
        stage_inputs, mapping = data_manager.prepare_input(stage_inputs, stage)

        stage_actions = state.get_stage_actions(stage)
        stage_outputs = [
            stage_actions[mapping[i][0]][mapping[i][1]][mapping[i][2]]
            for i in range(len(mapping))
        ]

        metrics = {}

        # Prompts
        processed_inputs = self._process_inputs(stage_inputs, for_training=True)
        prompt_ids = processed_inputs.input_ids.to(self.device)
        prompt_mask = processed_inputs.attention_mask.to(self.device)

        # Completions
        processed_outputs = self._process_inputs(stage_outputs, with_template=False, for_training=True)
        completion_ids = processed_outputs.input_ids.to(self.device)
        completion_mask = processed_outputs.attention_mask.to(self.device)

        model_inputs = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
        }

        # Rewards
        rewards = reward_manager[stage]
        rewards = [
            rewards[mapping[i][0]][mapping[i][1]][mapping[i][2]]
            for i in range(len(mapping))
        ]
        rewards = torch.tensor(rewards).to(self.device)

        # Advantages
        with torch.no_grad():
            advantages = rewards - rewards.mean(dim=1, keepdim=True)
            if rewards.shape[1] > 1:
                advantages /= rewards.std(dim=1, keepdim=True) + 1e-8
        model_inputs["advantages"] = advantages.flatten()

        # Loss
        loss = self.compute_loss(self.model, model_inputs)
        loss.backward()
        self.optimizer.step()
        self.model.zero_grad()

        metrics["train/loss"] = loss.item()
        metrics["train/rewards"] = rewards.mean().item()
        self.log(metrics, global_step)

        self.cleanup_step()
        return global_step

    # ----------------------------------------------------
    # MEMORY CLEANUP
    # ----------------------------------------------------
    def cleanup_step(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def cleanup(self):
        self.cleanup_trackers()
