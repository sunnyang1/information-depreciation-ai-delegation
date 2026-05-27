#!/usr/bin/env python3
"""
================================================================================
Production-Grade LLM Experiments for
"Information Depreciation and Optimal Depth in AI Delegation Chains"
================================================================================

Three experiments to validate theoretical predictions:
  1. Depth-Accuracy Tradeoff: Measure F1 as function of chain depth L
  2. Front-Loading Validation: Compare uniform vs front-loaded vs back-loaded
  3. Information Depreciation Rate (eta): Estimate per-layer retention factor

Supports:
  - Real LLM inference via transformers / vLLM
  - 4-bit quantization via bitsandbytes
  - Multi-GPU via device_map="auto"
  - Checkpoint/resume for long-running experiments
  - Detailed logging and result export (JSON, CSV, LaTeX)

Usage:
    # Basic run (single GPU)
    python run_real_experiments.py --experiment all --model_size 7b

    # Use vLLM acceleration
    python run_real_experiments.py --experiment all --use_vllm --model_size 7b

    # 4-bit quantization (fit large models on limited VRAM)
    python run_real_experiments.py --experiment all --quantization 4bit --model_size 13b

    # Only experiment 1
    python run_real_experiments.py --experiment depth --model_size 7b

    # Resume from checkpoint
    python run_real_experiments.py --experiment all --model_size 7b --resume

================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import csv
import time
import random
import logging
import argparse
import gc
import re
import math
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from enum import Enum
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional dependencies -- handle gracefully
# ---------------------------------------------------------------------------
try:
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
        BitsAndBytesConfig,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers not installed. Real LLM inference unavailable.")

try:
    import datasets
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    warnings.warn("datasets not installed. Dataset loading unavailable.")

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

try:
    from scipy.optimize import curve_fit
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not installed. Curve fitting will use numpy.")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm_experiments")

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Model registry: name -> {hf_id, params_B, context_window}
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "7b": {
        "hf_id": "meta-llama/Llama-2-7b-chat-hf",
        "params_B": 7,
        "context_window": 4096,
        "fallback_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Llama-2 7B Chat (small, fast)",
    },
    "13b": {
        "hf_id": "meta-llama/Llama-2-13b-chat-hf",
        "params_B": 13,
        "context_window": 4096,
        "fallback_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Llama-2 13B Chat (medium)",
    },
    "8b-instruct": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "params_B": 8,
        "context_window": 128_000,
        "fallback_id": "meta-llama/Llama-3-8B-Instruct",
        "description": "Llama-3.1 8B Instruct (recommended, best context)",
    },
    "70b": {
        "hf_id": "meta-llama/Llama-2-70b-chat-hf",
        "params_B": 70,
        "context_window": 4096,
        "fallback_id": None,
        "description": "Llama-2 70B Chat (large, needs 2+ GPUs)",
    },
    "tiny": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "params_B": 1.1,
        "context_window": 2048,
        "fallback_id": None,
        "description": "TinyLlama 1.1B (minimal, for testing)",
    },
    "phi2": {
        "hf_id": "microsoft/phi-2",
        "params_B": 2.7,
        "context_window": 2048,
        "fallback_id": None,
        # NOTE: Phi-2 requires transformers>=4.36.0 for native architecture support.
        # With transformers==4.35.0, use trust_remote_code=True (already set in loader)
        # but may still fail if the model config uses newer features.
        "description": "Microsoft Phi-2 (small, good quality). Requires transformers>=4.36 or trust_remote_code",
    },
}

# Experiment defaults
DEFAULT_TOTAL_BUDGET = 200_000  # tokens
DEFAULT_MAX_DEPTH = 5
DEFAULT_N_TRIALS = 100

# Needle-in-Haystack defaults
DEFAULT_N_FACTS = 100
NEEDLE_TEMPLATE = "The secret code of {entity} is {code}."
NEEDLE_ENTITIES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zack", "Anna", "Ben", "Clara", "David", "Emma", "Finn",
    "Gina", "Hugo", "Isla", "James", "Kara", "Liam", "Nora", "Oscar",
    "Pia", "Ryan", "Sara", "Tom", "Una", "Vince", "Will", "Zoe",
    "Aaron", "Bella", "Caleb", "Daisy", "Ethan", "Fiona", "George", "Hannah",
    "Ian", "Julia", "Kevin", "Luna", "Max", "Nina", "Owen", "Penny",
    "Quentin", "Rose", "Sean", "Tara", "Uri", "Vera", "Wade", "Xenia",
    "Yvonne", "Zane", "Adam", "Beth", "Carl", "Dana", "Eric", "Faith",
    "Gavin", "Hope", "Ivan", "Jade", "Kyle", "Lily", "Mark", "Nell",
    "Otto", "Page", "Reed", "Sage", "Troy", "Val", "Walt",
]

# Quantization config map
QUANT_CONFIGS = {
    "4bit": {"load_in_4bit": True, "bnb_4bit_compute_dtype": "float16"},
    "8bit": {"load_in_8bit": True},
}


# =============================================================================
# Budget allocation strategies
# =============================================================================

class BudgetStrategy(str, Enum):
    """Budget allocation strategy across chain layers."""
    UNIFORM = "uniform"
    FRONT_LOADED = "front_loaded"
    BACK_LOADED = "back_loaded"
    GEOMETRIC_FRONT = "geometric_front"
    GEOMETRIC_BACK = "geometric_back"


def allocate_budget(
    total_budget: int,
    depth: int,
    strategy: BudgetStrategy,
) -> List[int]:
    """
    Allocate context budget across L+1 layers according to strategy.

    Args:
        total_budget: Total token budget.
        depth: Number of chain layers (L).
        strategy: Allocation strategy.

    Returns:
        List of context sizes per layer (length = depth + 1).
    """
    n_layers = depth + 1
    if strategy == BudgetStrategy.UNIFORM:
        base = total_budget // n_layers
        sizes = [base] * n_layers
        # Distribute remainder
        for i in range(total_budget - sum(sizes)):
            sizes[i] += 1
    elif strategy == BudgetStrategy.FRONT_LOADED:
        # Geometric decay: layer 0 gets the most
        weights = np.array([2.0 ** (-i) for i in range(n_layers)])
        weights = weights / weights.sum()
        sizes = (weights * total_budget).astype(int).tolist()
        # Fix rounding
        diff = total_budget - sum(sizes)
        sizes[0] += diff
    elif strategy == BudgetStrategy.BACK_LOADED:
        # Reverse: last layer gets the most
        weights = np.array([2.0 ** (-(depth - i)) for i in range(n_layers)])
        weights = weights / weights.sum()
        sizes = (weights * total_budget).astype(int).tolist()
        diff = total_budget - sum(sizes)
        sizes[-1] += diff
    elif strategy == BudgetStrategy.GEOMETRIC_FRONT:
        # Halving each layer: A, A/2, A/4, ...
        weights = np.array([1.0 / (2 ** i) for i in range(n_layers)])
        weights = weights / weights.sum()
        sizes = (weights * total_budget).astype(int).tolist()
        diff = total_budget - sum(sizes)
        sizes[0] += diff
    elif strategy == BudgetStrategy.GEOMETRIC_BACK:
        weights = np.array([1.0 / (2 ** (depth - i)) for i in range(n_layers)])
        weights = weights / weights.sum()
        sizes = (weights * total_budget).astype(int).tolist()
        diff = total_budget - sum(sizes)
        sizes[-1] += diff
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return sizes


# =============================================================================
# Data classes for experiment results
# =============================================================================

@dataclass
class InferenceMetrics:
    """Metrics from a single inference call."""
    latency_ms: float
    input_tokens: int
    output_tokens: int
    peak_memory_mb: float
    start_time: float
    end_time: float


@dataclass
class ChainLayerResult:
    """Result from a single chain layer."""
    layer_idx: int
    model_name: str
    context_size: int
    input_text: str
    output_text: str
    metrics: InferenceMetrics


@dataclass
class ChainResult:
    """Result from running a full chain."""
    depth: int
    strategy: str
    total_budget: int
    layer_results: List[ChainLayerResult]
    final_output: str
    total_latency_ms: float
    total_tokens_used: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Aggregated experiment result."""
    experiment_id: str
    depth: int
    strategy: Optional[str]
    metric_value: float  # F1 score or accuracy
    metric_std: float
    precision_retained: float
    tokens_used: int
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DepreciationEstimate:
    """Estimated information retention at a given layer."""
    layer: int
    retention_rate: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    theoretical: float


@dataclass
class NeedleFact:
    """A single needle fact for needle-in-haystack experiments."""
    entity: str
    code: str
    full_text: str
    layer_inserted: int


# =============================================================================
# 1. ModelManager: Manage LLM loading and inference
# =============================================================================

class ModelManager:
    """
    Manages loading and inference for LLM models.

    Supports:
        - Loading from HuggingFace via transformers
        - vLLM accelerated inference (optional)
        - 4-bit / 8-bit quantization via bitsandbytes
        - Multi-GPU sharding via device_map="auto"
        - Automatic fallback to smaller models if primary unavailable

    Example:
        mm = ModelManager(model_size="7b", use_vllm=False, quantization="4bit")
        model = mm.load_model()
        output = model.generate("What is the capital of France?")
    """

    def __init__(
        self,
        model_size: str = "7b",
        use_vllm: bool = False,
        quantization: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        fallback: bool = True,
    ):
        """
        Initialize ModelManager.

        Args:
            model_size: Key from MODEL_REGISTRY ("7b", "13b", "8b-instruct", etc.).
            use_vllm: Whether to use vLLM for accelerated inference.
            quantization: "4bit", "8bit", or None for full precision.
            device: torch device string, or None for auto.
            cache_dir: HuggingFace cache directory.
            fallback: If True, falls back to smaller model on load failure.
        """
        if model_size not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_size '{model_size}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        if use_vllm and not HAS_VLLM:
            logger.warning("vLLM not installed, falling back to transformers")
            use_vllm = False
        if quantization and not HAS_BITSANDBYTES:
            logger.warning("bitsandbytes not installed, disabling quantization")
            quantization = None

        self.model_size = model_size
        self.use_vllm = use_vllm
        self.quantization = quantization
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.fallback = fallback

        self.config = MODEL_REGISTRY[model_size]
        self.model_id = self.config["hf_id"]
        self.context_window = self.config["context_window"]

        # Internal state
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._vllm_model: Optional[Any] = None
        self._loaded = False
        self._load_time: float = 0.0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load(self) -> "ModelManager":
        """Load the model and tokenizer. Returns self for chaining."""
        if self._loaded:
            return self

        logger.info("=" * 60)
        logger.info(f"Loading model: {self.model_id}")
        logger.info(f"  Size: {self.config['params_B']}B params")
        logger.info(f"  Context: {self.context_window}")
        logger.info(f"  vLLM: {self.use_vllm}")
        logger.info(f"  Quantization: {self.quantization}")
        logger.info(f"  Device: {self.device}")

        t0 = time.time()
        try:
            if self.use_vllm:
                self._load_vllm()
            else:
                self._load_transformers()
            self._load_time = time.time() - t0
            self._loaded = True
            logger.info(f"Model loaded in {self._load_time:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load {self.model_id}: {e}")
            if self.fallback and self.config["fallback_id"]:
                fallback_id = self.config["fallback_id"]
                logger.info(f"Attempting fallback to {fallback_id}")
                self.model_id = fallback_id
                return self.load()
            raise

        return self

    def _load_transformers(self) -> None:
        """Load model via transformers."""
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers not installed")

        tokenizer_kwargs = {"trust_remote_code": True}
        if self.cache_dir:
            tokenizer_kwargs["cache_dir"] = self.cache_dir

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            **tokenizer_kwargs,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Build quantization config
        quantization_config = None
        if self.quantization and HAS_BITSANDBYTES:
            qcfg = QUANT_CONFIGS.get(self.quantization, {})
            if self.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.quantization is None else "auto",
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["device_map"] = "auto"

        if self.cache_dir:
            model_kwargs["cache_dir"] = self.cache_dir

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

        if torch.cuda.device_count() == 1 and self.quantization is None:
            self._model = self._model.to(self.device)

        self._model.eval()
        logger.info(
            f"  Model on: {next(self._model.parameters()).device}"
        )
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            logger.info(f"  GPU memory: {mem_mb:.0f} MB")

    def _load_vllm(self) -> None:
        """Load model via vLLM (faster inference)."""
        gpu_count = torch.cuda.device_count()
        self._vllm_model = LLM(
            model=self.model_id,
            tensor_parallel_size=max(1, gpu_count),
            trust_remote_code=True,
            dtype="float16",
            download_dir=self.cache_dir,
        )
        logger.info(f"  vLLM loaded with tensor_parallel={max(1, gpu_count)}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None,
    ) -> Tuple[str, InferenceMetrics]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt string.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            do_sample: Whether to sample (False = greedy).
            stop_sequences: Optional list of strings to stop at.

        Returns:
            (generated_text, metrics)
        """
        if not self._loaded:
            self.load()

        t_start = time.time()
        mem_before = (
            torch.cuda.memory_allocated() / 1024 ** 2
            if torch.cuda.is_available()
            else 0.0
        )

        if self.use_vllm:
            text, in_tok, out_tok = self._generate_vllm(
                prompt, max_new_tokens, temperature, top_p, stop_sequences
            )
        else:
            text, in_tok, out_tok = self._generate_transformers(
                prompt, max_new_tokens, temperature, top_p, do_sample, stop_sequences
            )

        t_end = time.time()
        latency = (t_end - t_start) * 1000  # ms
        mem_after = (
            torch.cuda.max_memory_allocated() / 1024 ** 2
            if torch.cuda.is_available()
            else 0.0
        )

        metrics = InferenceMetrics(
            latency_ms=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            peak_memory_mb=mem_after - mem_before,
            start_time=t_start,
            end_time=t_end,
        )
        return text, metrics

    def _generate_transformers(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        stop_sequences: Optional[List[str]],
    ) -> Tuple[str, int, int]:
        """Generate via transformers."""
        assert self._tokenizer is not None and self._model is not None

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_window - max_new_tokens,
        )
        input_ids = inputs["input_ids"].to(self.device)
        in_tokens = input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if stop_sequences:
            # Build stopping criteria from strings
            from transformers import StoppingCriteria, StoppingCriteriaList

            class StopOnStrings(StoppingCriteria):
                def __init__(self, tokenizer, stop_strings, device):
                    self.tokenizer = tokenizer
                    self.stop_strings = stop_strings
                    self.device = device

                def __call__(self, input_ids, scores, **kwargs):
                    generated = self.tokenizer.decode(
                        input_ids[0], skip_special_tokens=True
                    )
                    for s in self.stop_strings:
                        if s in generated:
                            return True
                    return False

            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                StopOnStrings(self._tokenizer, stop_sequences, self.device)
            ])

        with torch.no_grad():
            outputs = self._model.generate(input_ids, **gen_kwargs)

        out_tokens = outputs.shape[1] - in_tokens
        generated_text = self._tokenizer.decode(
            outputs[0][in_tokens:], skip_special_tokens=True
        )
        return generated_text.strip(), in_tokens, out_tokens

    def _generate_vllm(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]],
    ) -> Tuple[str, int, int]:
        """Generate via vLLM."""
        assert self._vllm_model is not None

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop_sequences or [],
        )

        outputs = self._vllm_model.generate([prompt], sampling_params)
        output = outputs[0]

        generated_text = output.outputs[0].text.strip()
        in_tokens = len(output.prompt_token_ids)
        out_tokens = len(output.outputs[0].token_ids)
        return generated_text, in_tokens, out_tokens

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 8,
    ) -> List[Tuple[str, InferenceMetrics]]:
        """
        Generate for a batch of prompts.

        Args:
            prompts: List of prompt strings.
            max_new_tokens: Max tokens per generation.
            temperature: Sampling temperature.
            top_p: Nucleus threshold.
            batch_size: Batch size for processing.

        Returns:
            List of (text, metrics) tuples.
        """
        if not self._loaded:
            self.load()

        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            if self.use_vllm:
                batch_results = self._generate_batch_vllm(
                    batch, max_new_tokens, temperature, top_p
                )
            else:
                batch_results = self._generate_batch_transformers(
                    batch, max_new_tokens, temperature, top_p
                )
            results.extend(batch_results)
        return results

    def _generate_batch_vllm(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[Tuple[str, InferenceMetrics]]:
        """Batch generation via vLLM."""
        t_start = time.time()
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self._vllm_model.generate(prompts, sampling_params)
        t_end = time.time()

        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            in_tok = len(output.prompt_token_ids)
            out_tok = len(output.outputs[0].token_ids)
            metrics = InferenceMetrics(
                latency_ms=(t_end - t_start) * 1000 / len(prompts),
                input_tokens=in_tok,
                output_tokens=out_tok,
                peak_memory_mb=0.0,
                start_time=t_start,
                end_time=t_end,
            )
            results.append((text, metrics))
        return results

    def _generate_batch_transformers(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[Tuple[str, InferenceMetrics]]:
        """Batch generation via transformers (manual batching)."""
        t_start = time.time()
        max_prompt_len = (
            self.context_window - max_new_tokens
        )

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        t_end = time.time()
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()

        results = []
        for i, output_ids in enumerate(outputs):
            out_tokens = output_ids.shape[0] - input_lens[i]
            generated = self._tokenizer.decode(
                output_ids[input_lens[i] :], skip_special_tokens=True
            ).strip()
            metrics = InferenceMetrics(
                latency_ms=(t_end - t_start) * 1000 / len(prompts),
                input_tokens=input_lens[i],
                output_tokens=out_tokens,
                peak_memory_mb=0.0,
                start_time=t_start,
                end_time=t_end,
            )
            results.append((generated, metrics))
        return results

    # ------------------------------------------------------------------
    # Truncation helper
    # ------------------------------------------------------------------
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens."""
        if not self._tokenizer:
            # Rough approximation: ~4 chars per token
            approx_chars = max_tokens * 4
            return text[:approx_chars]
        tokens = self._tokenizer.encode(text, truncation=False)
        if len(tokens) <= max_tokens:
            return text
        truncated = self._tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
        return truncated

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer is None:
            return len(text.split())
        return len(self._tokenizer.encode(text))

    def unload(self) -> None:
        """Free model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._vllm_model is not None:
            del self._vllm_model
            self._vllm_model = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded from memory")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return (
            f"ModelManager({self.model_id}, "
            f"vllm={self.use_vllm}, quant={self.quantization})"
        )


# =============================================================================
# 2. DatasetLoader: Load SQuAD, HotpotQA, GSM8K, Needle-in-Haystack
# =============================================================================

class DatasetLoader:
    """
    Loads and preprocesses datasets for experiments.

    Supported datasets:
        - SQuAD v2 (reading comprehension)
        - HotpotQA (multi-hop reasoning)
        - GSM8K (math reasoning)
        - Needle-in-Haystack (custom generated)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._squad = None
        self._hotpotqa = None
        self._gsm8k = None

    # ------------------------------------------------------------------
    # SQuAD
    # ------------------------------------------------------------------
    def load_squad(self, split: str = "validation", max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Load SQuAD v2 dataset.

        Args:
            split: "train" or "validation".
            max_samples: Limit number of samples.

        Returns:
            List of dicts with keys: id, context, question, answers.
        """
        logger.info(f"Loading SQuAD v2 ({split})...")
        if HAS_DATASETS:
            try:
                ds = load_dataset("squad_v2", split=split, cache_dir=self.cache_dir)
            except Exception as e:
                logger.warning(f"Failed to load squad_v2: {e}, using mock data")
                return self._mock_squad(max_samples or 50)
        else:
            logger.warning("datasets library not available, using mock data")
            return self._mock_squad(max_samples or 50)

        results = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            answers = item.get("answers", {})
            answer_texts = answers.get("text", [])
            results.append({
                "id": item.get("id", f"squad_{i}"),
                "context": item["context"],
                "question": item["question"],
                "answers": answer_texts if answer_texts else [""],
            })
        logger.info(f"  Loaded {len(results)} SQuAD samples")
        return results

    def _mock_squad(self, n: int = 50) -> List[Dict[str, str]]:
        """Generate mock SQuAD-style data for testing."""
        contexts = [
            "The University of Notre Dame began late on the bitterly cold afternoon of November 26, 1842, when Father Edward Sorin, C.S.C., a 28-year-old French priest, arrived at the Snow Fields along the St. Joseph River in Indiana.",
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
            "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities.",
            "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups.",
            "Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.",
        ]
        questions = [
            "When did the University of Notre Dame begin?",
            "Who is the Eiffel Tower named after?",
            "What is photosynthesis?",
            "What is the Great Wall of China?",
            "What is machine learning?",
        ]
        answers = [
            ["November 26, 1842"],
            ["Gustave Eiffel"],
            ["a process used by plants"],
            ["a series of fortifications"],
            ["a branch of artificial intelligence"],
        ]
        results = []
        for i in range(n):
            idx = i % len(contexts)
            results.append({
                "id": f"mock_squad_{i}",
                "context": contexts[idx],
                "question": questions[idx],
                "answers": answers[idx],
            })
        logger.info(f"  Generated {len(results)} mock SQuAD samples")
        return results

    # ------------------------------------------------------------------
    # HotpotQA
    # ------------------------------------------------------------------
    def load_hotpotqa(self, split: str = "validation", max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Load HotpotQA (multi-hop reasoning).

        Args:
            split: "train" or "validation".
            max_samples: Limit samples.

        Returns:
            List of dicts with: id, context, question, answer.
        """
        logger.info(f"Loading HotpotQA ({split})...")
        if HAS_DATASETS:
            try:
                ds = load_dataset("hotpot_qa", "distractor", split=split, cache_dir=self.cache_dir)
            except Exception as e:
                logger.warning(f"Failed to load hotpot_qa: {e}, using mock data")
                return self._mock_hotpotqa(max_samples or 50)
        else:
            return self._mock_hotpotqa(max_samples or 50)

        results = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            # Flatten context paragraphs
            paragraphs = item.get("context", {}).get("sentences", [])
            flat = " ".join([" ".join(sents) for sents in paragraphs])
            results.append({
                "id": item.get("_id", f"hotpot_{i}"),
                "context": flat,
                "question": item["question"],
                "answer": item.get("answer", ""),
            })
        logger.info(f"  Loaded {len(results)} HotpotQA samples")
        return results

    def _mock_hotpotqa(self, n: int = 50) -> List[Dict[str, str]]:
        """Generate mock HotpotQA data."""
        data = [
            {
                "context": "The Statue of Liberty was a gift from France to the United States. It was designed by French sculptor Frederic Auguste Bartholdi. Gustave Eiffel designed the internal structure. The statue was dedicated on October 28, 1886.",
                "question": "Who designed the internal structure of the Statue of Liberty and what country gifted it?",
                "answer": "Gustave Eiffel; France",
            },
            {
                "context": "Marie Curie was a Polish physicist and chemist. She won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911. She was the first woman to win a Nobel Prize.",
                "question": "What prizes did Marie Curie win and when?",
                "answer": "Nobel Prize in Physics 1903, Nobel Prize in Chemistry 1911",
            },
            {
                "context": "The Amazon River is in South America. It is the largest river by discharge volume of water in the world. It flows through Brazil, Peru, and Colombia. The Amazon rainforest surrounds it.",
                "question": "Which countries does the Amazon River flow through and what surrounds it?",
                "answer": "Brazil, Peru, Colombia; Amazon rainforest",
            },
        ]
        results = []
        for i in range(n):
            item = data[i % len(data)]
            results.append({
                "id": f"mock_hotpot_{i}",
                "context": item["context"],
                "question": item["question"],
                "answer": item["answer"],
            })
        logger.info(f"  Generated {len(results)} mock HotpotQA samples")
        return results

    # ------------------------------------------------------------------
    # GSM8K
    # ------------------------------------------------------------------
    def load_gsm8k(self, split: str = "test", max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """Load GSM8K math problems."""
        logger.info(f"Loading GSM8K ({split})...")
        if HAS_DATASETS:
            try:
                ds = load_dataset("gsm8k", "main", split=split, cache_dir=self.cache_dir)
            except Exception as e:
                logger.warning(f"Failed to load gsm8k: {e}, using mock")
                return self._mock_gsm8k(max_samples or 50)
        else:
            return self._mock_gsm8k(max_samples or 50)

        results = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            answer = item.get("answer", "")
            # Extract final number
            final_answer = answer.split("####")[-1].strip() if "####" in answer else answer
            results.append({
                "id": f"gsm8k_{i}",
                "question": item["question"],
                "answer": final_answer,
                "full_answer": answer,
            })
        logger.info(f"  Loaded {len(results)} GSM8K samples")
        return results

    def _mock_gsm8k(self, n: int = 50) -> List[Dict[str, str]]:
        problems = [
            {"question": "Janet has 24 ducks. She buys 5 more. How many ducks does she have?", "answer": "29"},
            {"question": "A train travels 60 miles per hour for 3 hours. How far does it go?", "answer": "180"},
            {"question": "Apples cost $2 each. Bananas cost $1 each. If you buy 3 apples and 4 bananas, how much do you spend?", "answer": "10"},
        ]
        return [{"id": f"mock_gsm8k_{i}", **problems[i % len(problems)]} for i in range(n)]

    # ------------------------------------------------------------------
    # Needle-in-Haystack generation
    # ------------------------------------------------------------------
    def generate_needle_dataset(
        self,
        n_facts: int = DEFAULT_N_FACTS,
        base_context: Optional[str] = None,
        entities: Optional[List[str]] = None,
    ) -> Tuple[str, List[NeedleFact]]:
        """
        Generate a Needle-in-Haystack dataset.

        Args:
            n_facts: Number of needle facts to insert.
            base_context: Base text to insert facts into (or auto-generate).
            entities: List of entity names to use.

        Returns:
            (context_with_facts, list_of_NeedleFact)
        """
        entities = entities or NEEDLE_ENTITIES[:n_facts]
        n_facts = min(n_facts, len(entities))

        facts: List[NeedleFact] = []
        fact_texts: List[str] = []
        for i, entity in enumerate(entities[:n_facts]):
            code = f"CODE-{random.randint(1000, 9999)}-{i}"
            text = NEEDLE_TEMPLATE.format(entity=entity, code=code)
            facts.append(NeedleFact(
                entity=entity,
                code=code,
                full_text=text,
                layer_inserted=0,
            ))
            fact_texts.append(text)

        if base_context is None:
            # Generate filler text
            filler_paragraphs = [
                "The history of human civilization spans thousands of years.",
                "Ancient Egypt built pyramids that still stand today as monuments to their engineering prowess.",
                "The Roman Empire once controlled vast territories across Europe, Asia, and Africa.",
                "During the Renaissance, artists like Leonardo da Vinci and Michelangelo created masterpieces.",
                "The Industrial Revolution transformed society with new manufacturing processes.",
                "In the 20th century, two world wars reshaped the geopolitical landscape.",
                "The invention of the transistor led to the digital revolution we experience today.",
                "Modern science continues to push the boundaries of human knowledge.",
                "Climate change poses significant challenges for future generations.",
                "Space exploration has taken humans to the Moon and sent probes beyond our solar system.",
                "Artificial intelligence represents one of the most transformative technologies of our era.",
                "Global trade networks connect economies across all continents.",
                "Education remains a cornerstone of societal progress and individual opportunity.",
                "Medical advances have dramatically increased human lifespan over the past century.",
                "Renewable energy sources are becoming increasingly competitive with fossil fuels.",
            ]
            # Build long context by repeating and shuffling filler
            random.shuffle(filler_paragraphs)
            paragraphs = []
            for i in range(max(50, n_facts * 2)):
                paragraphs.append(filler_paragraphs[i % len(filler_paragraphs)])
                if i % 3 == 0 and fact_texts:
                    paragraphs.append(fact_texts.pop(0))
            while fact_texts:
                # Insert remaining facts at random positions
                pos = random.randint(0, len(paragraphs))
                paragraphs.insert(pos, fact_texts.pop(0))
            base_context = "\n\n".join(paragraphs)
        else:
            # Insert facts into provided context
            sentences = base_context.split(".")
            for fact in fact_texts:
                pos = random.randint(0, max(1, len(sentences) - 1))
                sentences.insert(pos, fact)
            base_context = ". ".join(sentences)

        return base_context, facts

    def get_needle_questions(self, facts: List[NeedleFact]) -> List[Dict[str, str]]:
        """Generate questions to test fact retrieval."""
        questions = []
        for fact in facts:
            questions.append({
                "id": f"needle_{fact.entity}",
                "question": f"What is the secret code of {fact.entity}?",
                "answer": fact.code,
                "entity": fact.entity,
            })
        return questions


# =============================================================================
# 3. LLMChain: Multi-layer LLM inference chain
# =============================================================================

class LLMChain:
    """
    Multi-layer LLM inference chain.

    Each layer processes the output of the previous layer, with configurable
    context window sizes and budget allocation strategies.

    Example:
        chain = LLMChain(models={0: model_mgr_7b, 1: model_mgr_7b})
        result = chain.run(context, question, depth=2, strategy=BudgetStrategy.UNIFORM)
    """

    def __init__(
        self,
        models: Dict[int, ModelManager],
        context_sizes: Optional[Dict[int, int]] = None,
    ):
        """
        Args:
            models: Mapping from layer index to ModelManager.
            context_sizes: Optional mapping from layer index to max context tokens.
                           If None, uses each model's default context_window.
        """
        self.models = models
        self.context_sizes = context_sizes or {}
        self.layer_history: List[ChainLayerResult] = []

    def _get_model_for_layer(self, layer_idx: int) -> ModelManager:
        """Get the model manager for a layer (wraps if needed)."""
        max_layer = max(self.models.keys())
        actual_key = min(layer_idx, max_layer)
        return self.models[actual_key]

    def _get_context_size(self, layer_idx: int, allocated: int) -> int:
        """Get effective context size for a layer."""
        model = self._get_model_for_layer(layer_idx)
        max_ctx = self.context_sizes.get(layer_idx, model.context_window)
        return min(allocated, max_ctx)

    def _build_prompt(self, context: str, question: str, layer_idx: int) -> str:
        """Build a prompt for a chain layer."""
        if layer_idx == 0:
            # First layer: direct question answering
            prompt = (
                f"Read the following passage carefully and answer the question.\n\n"
                f"Passage: {context}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
        else:
            # Subsequent layers: refine previous answer
            prompt = (
                f"A previous analysis produced the following intermediate result:\n"
                f"---\n{context}\n---\n\n"
                f"Based on this, answer: {question}\n\n"
                f"Provide a clear, concise answer. If the previous analysis "
                f"contains the answer, extract it directly. "
                f"Answer:"
            )
        return prompt

    def run(
        self,
        context: str,
        question: str,
        depth: int,
        strategy: BudgetStrategy,
        total_budget: int = DEFAULT_TOTAL_BUDGET,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ) -> ChainResult:
        """
        Run the multi-layer chain.

        Args:
            context: Input context/passage.
            question: Question to answer.
            depth: Number of chain layers (L).
            strategy: Budget allocation strategy.
            total_budget: Total token budget.
            max_new_tokens: Max tokens per generation.
            temperature: Sampling temperature.

        Returns:
            ChainResult with all layer outputs and metrics.
        """
        contexts = allocate_budget(total_budget, depth, strategy)
        self.layer_history = []
        t_start = time.time()
        current_context = context
        total_tokens = 0

        logger.debug(f"Running chain: depth={depth}, strategy={strategy.value}")
        logger.debug(f"  Context allocation: {contexts}")

        for layer_idx in range(depth + 1):
            model = self._get_model_for_layer(layer_idx)
            if not model.is_loaded:
                model.load()

            ctx_size = self._get_context_size(layer_idx, contexts[layer_idx])
            # Truncate context to fit
            truncated = model.truncate_to_tokens(current_context, ctx_size)

            prompt = self._build_prompt(truncated, question, layer_idx)
            output, metrics = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=(temperature > 0),
            )

            total_tokens += metrics.input_tokens + metrics.output_tokens
            layer_result = ChainLayerResult(
                layer_idx=layer_idx,
                model_name=model.model_id,
                context_size=ctx_size,
                input_text=truncated[:200] + "..." if len(truncated) > 200 else truncated,
                output_text=output,
                metrics=metrics,
            )
            self.layer_history.append(layer_result)

            logger.debug(
                f"  Layer {layer_idx}: {metrics.input_tokens} -> "
                f"{metrics.output_tokens} tokens, "
                f"{metrics.latency_ms:.0f}ms"
            )

            # Next layer's context is this layer's output
            current_context = output

        t_end = time.time()
        return ChainResult(
            depth=depth,
            strategy=strategy.value,
            total_budget=total_budget,
            layer_results=self.layer_history,
            final_output=current_context,
            total_latency_ms=(t_end - t_start) * 1000,
            total_tokens_used=total_tokens,
            metadata={"context_allocation": contexts},
        )

    def run_batch(
        self,
        examples: List[Dict[str, str]],
        depth: int,
        strategy: BudgetStrategy,
        total_budget: int = DEFAULT_TOTAL_BUDGET,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ) -> List[ChainResult]:
        """Run chain on a batch of examples (sequentially)."""
        results = []
        for i, ex in enumerate(examples):
            logger.debug(f"Batch item {i + 1}/{len(examples)}")
            result = self.run(
                context=ex["context"],
                question=ex["question"],
                depth=depth,
                strategy=strategy,
                total_budget=total_budget,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results.append(result)
        return results



# =============================================================================
# 4. ExperimentRunner: Run all 3 experiments
# =============================================================================

class ExperimentRunner:
    """
    Runs the three experiments for the paper.

    Experiments:
        1. Depth-Accuracy Tradeoff: Vary chain depth L, measure F1
        2. Front-Loading Validation: Compare budget allocation strategies
        3. Information Depreciation Rate (eta): Needle-in-Haystack

    Supports checkpoint/resume for long-running experiments.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        output_dir: str = str(DEFAULT_OUTPUT_DIR),
        checkpoint_dir: str = str(CHECKPOINT_DIR),
    ):
        """
        Args:
            model_manager: Primary model manager for inference.
            output_dir: Directory to save results.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model_manager = model_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_loader = DatasetLoader()
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def _checkpoint_path(self, experiment: str) -> Path:
        return self.checkpoint_dir / f"{experiment}_checkpoint.json"

    def _save_checkpoint(self, experiment: str, data: Dict) -> None:
        """Save experiment checkpoint."""
        path = self._checkpoint_path(experiment)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, experiment: str) -> Optional[Dict]:
        """Load experiment checkpoint if exists."""
        path = self._checkpoint_path(experiment)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            logger.info(f"Resumed from checkpoint: {path}")
            return data
        return None

    # ------------------------------------------------------------------
    # Experiment 1: Depth-Accuracy Tradeoff
    # ------------------------------------------------------------------
    def run_depth_accuracy(
        self,
        depths: List[int] = None,
        total_budget: int = DEFAULT_TOTAL_BUDGET,
        max_samples: int = 200,
        resume: bool = False,
    ) -> List[ExperimentResult]:
        """
        Experiment 1: Measure F1 score as function of chain depth L.

        Prediction: F1 is hump-shaped or decreasing in L.
        (Proposition 6: Precision Path Characterization)

        Args:
            depths: List of chain depths to test (default [1,2,3,4,5]).
            total_budget: Total token budget (default 200K).
            max_samples: Number of SQuAD examples.
            resume: Resume from checkpoint.

        Returns:
            List of ExperimentResult, one per depth.
        """
        depths = depths or list(range(1, DEFAULT_MAX_DEPTH + 1))
        experiment_name = "depth_accuracy"

        logger.info("=" * 70)
        logger.info("EXPERIMENT 1: Depth-Accuracy Tradeoff")
        logger.info("=" * 70)
        logger.info(f"  Depths: {depths}")
        logger.info(f"  Budget: {total_budget:,} tokens")
        logger.info(f"  Samples: {max_samples}")

        # Resume check
        if resume:
            ckpt = self._load_checkpoint(experiment_name)
            if ckpt:
                completed_depths = ckpt.get("completed_depths", [])
                results = ckpt.get("results", [])
                logger.info(f"  Resumed: completed {completed_depths}")
            else:
                completed_depths = []
                results = []
        else:
            completed_depths = []
            results = []

        # Load dataset
        examples = self.dataset_loader.load_squad(
            split="validation", max_samples=max_samples
        )

        # Build chain (single model for all layers)
        models = {i: self.model_manager for i in range(max(depths) + 1)}
        chain = LLMChain(models=models)

        for depth in depths:
            if depth in completed_depths:
                logger.info(f"  Skipping depth={depth} (already done)")
                continue

            logger.info(f"\n  --- Running depth L={depth} ---")
            f1_scores = []
            latencies = []
            total_tokens = []

            for i, ex in enumerate(examples):
                if (i + 1) % 20 == 0:
                    logger.info(f"    Processed {i + 1}/{len(examples)}")

                try:
                    result = chain.run(
                        context=ex["context"],
                        question=ex["question"],
                        depth=depth,
                        strategy=BudgetStrategy.UNIFORM,
                        total_budget=total_budget,
                        max_new_tokens=128,
                        temperature=0.1,
                    )
                    f1 = compute_f1(result.final_output, ex["answers"])
                    f1_scores.append(f1)
                    latencies.append(result.total_latency_ms)
                    total_tokens.append(result.total_tokens_used)
                except Exception as e:
                    logger.error(f"    Error on example {i}: {e}")
                    continue

            mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
            std_f1 = float(np.std(f1_scores)) if f1_scores else 0.0
            avg_latency = float(np.mean(latencies)) if latencies else 0.0
            avg_tokens = float(np.mean(total_tokens)) if total_tokens else 0

            # Theoretical precision retained
            avg_context = total_budget // (depth + 1)
            eta_est = self._estimate_eta_from_context(avg_context)
            precision_retained = eta_est ** depth

            exp_result = ExperimentResult(
                experiment_id=f"depth_{depth}",
                depth=depth,
                strategy=BudgetStrategy.UNIFORM.value,
                metric_value=mean_f1,
                metric_std=std_f1,
                precision_retained=precision_retained,
                tokens_used=int(np.sum(total_tokens)) if total_tokens else 0,
                latency_ms=avg_latency,
                metadata={
                    "n_samples": len(f1_scores),
                    "f1_scores": f1_scores[:10],  # Save subset
                    "avg_context_per_layer": avg_context,
                    "eta_estimate": eta_est,
                },
            )
            results.append(exp_result)
            completed_depths.append(depth)

            logger.info(
                f"  Depth L={depth}: F1={mean_f1:.3f} (±{std_f1:.3f}), "
                f"Precision retained={precision_retained:.3f}"
            )

            # Save checkpoint after each depth
            self._save_checkpoint(experiment_name, {
                "completed_depths": completed_depths,
                "results": [asdict(r) for r in results],
            })

        # Statistical summary
        f1s = [r.metric_value for r in results]
        pattern = self._classify_pattern(f1s)
        logger.info(f"\n  Pattern across depths: {pattern}")

        self.results[experiment_name] = results
        return results

    # ------------------------------------------------------------------
    # Experiment 2: Front-Loading Validation
    # ------------------------------------------------------------------
    def run_front_loading(
        self,
        depth: int = 3,
        total_budget: int = DEFAULT_TOTAL_BUDGET,
        strategies: List[BudgetStrategy] = None,
        max_samples: int = 200,
        resume: bool = False,
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Experiment 2: Compare budget allocation strategies.

        Prediction: Front-loaded > Uniform > Back-loaded (Proposition 8).

        Args:
            depth: Chain depth (default 3).
            total_budget: Total token budget.
            strategies: List of strategies to compare.
            max_samples: Number of HotpotQA examples.
            resume: Resume from checkpoint.

        Returns:
            Dict mapping strategy name to list of results.
        """
        strategies = strategies or [
            BudgetStrategy.UNIFORM,
            BudgetStrategy.FRONT_LOADED,
            BudgetStrategy.BACK_LOADED,
        ]
        experiment_name = "front_loading"

        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT 2: Front-Loading Validation")
        logger.info("=" * 70)
        logger.info(f"  Depth: L={depth}")
        logger.info(f"  Budget: {total_budget:,} tokens")
        logger.info(f"  Strategies: {[s.value for s in strategies]}")

        # Load dataset
        examples = self.dataset_loader.load_hotpotqa(
            split="validation", max_samples=max_samples
        )

        models = {i: self.model_manager for i in range(depth + 1)}
        chain = LLMChain(models=models)

        all_results: Dict[str, List[ExperimentResult]] = {}

        for strategy in strategies:
            logger.info(f"\n  --- Strategy: {strategy.value} ---")

            # Resume check
            if resume:
                ckpt = self._load_checkpoint(f"{experiment_name}_{strategy.value}")
                if ckpt and ckpt.get("completed"):
                    logger.info(f"    Skipping {strategy.value} (completed)")
                    all_results[strategy.value] = [
                        ExperimentResult(**r) for r in ckpt["results"]
                    ]
                    continue

            f1_scores = []
            latencies = []

            for i, ex in enumerate(examples):
                if (i + 1) % 20 == 0:
                    logger.info(f"    Processed {i + 1}/{len(examples)}")

                try:
                    result = chain.run(
                        context=ex["context"],
                        question=ex["question"],
                        depth=depth,
                        strategy=strategy,
                        total_budget=total_budget,
                        max_new_tokens=128,
                        temperature=0.1,
                    )
                    f1 = compute_f1(result.final_output, [ex["answer"]])
                    f1_scores.append(f1)
                    latencies.append(result.total_latency_ms)
                except Exception as e:
                    logger.error(f"    Error on example {i}: {e}")
                    continue

            mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
            std_f1 = float(np.std(f1_scores)) if f1_scores else 0.0

            # Allocations for display
            allocations = allocate_budget(total_budget, depth, strategy)

            exp_result = ExperimentResult(
                experiment_id=f"front_loading_{strategy.value}",
                depth=depth,
                strategy=strategy.value,
                metric_value=mean_f1,
                metric_std=std_f1,
                precision_retained=0.0,  # Not primary metric here
                tokens_used=sum(allocations),
                latency_ms=float(np.mean(latencies)) if latencies else 0,
                metadata={
                    "n_samples": len(f1_scores),
                    "allocations": allocations,
                    "f1_scores": f1_scores[:10],
                },
            )
            all_results[strategy.value] = [exp_result]

            logger.info(
                f"  Strategy {strategy.value}: F1={mean_f1:.3f} (±{std_f1:.3f}), "
                f"Allocation={allocations}"
            )

            self._save_checkpoint(f"{experiment_name}_{strategy.value}", {
                "completed": True,
                "results": [asdict(exp_result)],
            })

        # Rank strategies
        means = {
            s: np.mean([r.metric_value for r in res])
            for s, res in all_results.items()
        }
        ranking = sorted(means.items(), key=lambda x: -x[1])
        logger.info(f"\n  Ranking: {' > '.join([f'{s}({m:.3f})' for s, m in ranking])}")

        pred_confirmed = (
            ranking[0][0] == BudgetStrategy.FRONT_LOADED.value and
            ranking[1][0] == BudgetStrategy.UNIFORM.value and
            ranking[2][0] == BudgetStrategy.BACK_LOADED.value
        )
        logger.info(f"  Prediction confirmed: {pred_confirmed}")

        self.results[experiment_name] = all_results
        return all_results

    # ------------------------------------------------------------------
    # Experiment 3: Eta Estimation via Needle-in-Haystack
    # ------------------------------------------------------------------
    def run_eta_estimation(
        self,
        n_facts: int = DEFAULT_N_FACTS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        resume: bool = False,
    ) -> List[DepreciationEstimate]:
        """
        Experiment 3: Estimate per-layer information depreciation rate eta.

        Method: Insert n identifiable facts, pass through L layers,
        measure retention at each layer. Fit exponential decay.

        Args:
            n_facts: Number of needle facts (default 100).
            max_depth: Maximum chain depth to test.
            resume: Resume from checkpoint.

        Returns:
            List of DepreciationEstimate per layer.
        """
        experiment_name = "eta_estimation"

        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT 3: Information Depreciation Rate (eta)")
        logger.info("=" * 70)
        logger.info(f"  Facts: {n_facts}")
        logger.info(f"  Max depth: {max_depth}")

        # Resume check
        if resume:
            ckpt = self._load_checkpoint(experiment_name)
            if ckpt and ckpt.get("completed"):
                logger.info("  Experiment already completed (checkpoint)")
                return [DepreciationEstimate(**e) for e in ckpt["estimates"]]

        # Generate needle dataset
        context, facts = self.dataset_loader.generate_needle_dataset(n_facts=n_facts)
        questions = self.dataset_loader.get_needle_questions(facts)

        logger.info(f"  Generated context: {len(context)} chars, {len(facts)} facts")

        # Build chain
        models = {i: self.model_manager for i in range(max_depth + 1)}
        chain = LLMChain(models=models)

        estimates: List[DepreciationEstimate] = []

        for depth in range(1, max_depth + 1):
            logger.info(f"\n  --- Testing depth L={depth} ---")

            # Count how many facts are retrievable after passing through 'depth' layers
            retained_count = 0
            total_q_time = 0.0

            for i, q in enumerate(questions):
                if (i + 1) % 20 == 0:
                    logger.info(f"    Tested {i + 1}/{len(questions)} facts...")

                try:
                    t0 = time.time()
                    result = chain.run(
                        context=context,
                        question=q["question"],
                        depth=depth,
                        strategy=BudgetStrategy.UNIFORM,
                        total_budget=DEFAULT_TOTAL_BUDGET,
                        max_new_tokens=64,
                        temperature=0.1,
                    )
                    total_q_time += (time.time() - t0) * 1000

                    # Check if answer contains the correct code
                    if q["answer"].lower() in result.final_output.lower():
                        retained_count += 1
                except Exception as e:
                    logger.error(f"    Error on fact {i}: {e}")
                    continue

            retention_rate = retained_count / n_facts
            se = math.sqrt(retention_rate * (1 - retention_rate) / n_facts)
            ci_lower = max(0.0, retention_rate - 1.96 * se)
            ci_upper = min(1.0, retention_rate + 1.96 * se)

            estimate = DepreciationEstimate(
                layer=depth,
                retention_rate=retention_rate,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n_samples=n_facts,
                theoretical=0.0,  # Filled later
            )
            estimates.append(estimate)

            logger.info(
                f"  Depth {depth}: Retention={retention_rate:.3f} "
                f"[{ci_lower:.3f}, {ci_upper:.3f}] "
                f"({retained_count}/{n_facts} facts)"
            )

        # Fit exponential decay: R(L) = eta^L
        layers = np.arange(1, len(estimates) + 1)
        retention_rates = np.array([e.retention_rate for e in estimates])

        eta_estimated, r_squared = fit_exponential_decay(layers, retention_rates)

        # Fill theoretical values
        for e in estimates:
            e.theoretical = eta_estimated ** e.layer

        logger.info(f"\n  Estimated eta = {eta_estimated:.3f}")
        logger.info(f"  R-squared of exponential fit: {r_squared:.3f}")

        self._save_checkpoint(experiment_name, {
            "completed": True,
            "estimates": [asdict(e) for e in estimates],
            "eta": eta_estimated,
            "r_squared": r_squared,
        })

        self.results[experiment_name] = estimates
        return estimates

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _estimate_eta_from_context(self, context_size: int) -> float:
        """Estimate eta from context size (endogenous model)."""
        eta_bar = 0.95
        b = 3.0
        max_ctx = 128_000
        relative = context_size / max_ctx
        eta = eta_bar * (1 - math.exp(-b * relative))
        return max(0.5, min(eta, 0.95))

    def _classify_pattern(self, values: List[float]) -> str:
        """Classify the pattern of values across depths."""
        if len(values) < 3:
            return "insufficient_data"
        # Hump-shaped: first rises then falls
        max_idx = values.index(max(values))
        if 0 < max_idx < len(values) - 1:
            return "hump_shaped"
        # Monotonically decreasing
        if all(values[i] >= values[i + 1] for i in range(len(values) - 1)):
            return "monotonically_decreasing"
        return "other"


# =============================================================================
# 5. ResultsAnalyzer: Analyze results, fit models, export LaTeX/JSON/CSV
# =============================================================================

class ResultsAnalyzer:
    """
    Analyzes experiment results and exports in multiple formats.

    Supports:
        - Statistical summaries (mean, std, confidence intervals)
        - Exponential decay fitting
        - LaTeX table generation
        - JSON and CSV export
    """

    def __init__(self, output_dir: str = str(DEFAULT_OUTPUT_DIR)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------
    @staticmethod
    def mean_std(values: List[float]) -> Tuple[float, float]:
        """Compute mean and standard deviation."""
        if not values:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.std(values))

    @staticmethod
    def confidence_interval(
        values: List[float], confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence interval for mean."""
        if not values:
            return 0.0, 0.0
        mean = np.mean(values)
        se = stats.sem(values) if HAS_SCIPY else np.std(values) / math.sqrt(len(values))
        if HAS_SCIPY:
            ci = stats.t.interval(
                confidence, len(values) - 1, loc=mean, scale=se
            )
            return ci if ci else (mean, mean)
        else:
            z = 1.96  # approximate for 95%
            return mean - z * se, mean + z * se

    # ------------------------------------------------------------------
    # Export to JSON
    # ------------------------------------------------------------------
    def export_json(self, results: Dict[str, Any], filename: str = "results.json") -> str:
        """Export results to JSON file."""
        path = self.output_dir / filename

        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=serialize)
        logger.info(f"Results saved to {path}")
        return str(path)

    # ------------------------------------------------------------------
    # Export to CSV
    # ------------------------------------------------------------------
    def export_csv(
        self,
        records: List[Dict[str, Any]],
        filename: str = "results.csv",
    ) -> str:
        """Export records to CSV file."""
        if not records:
            return ""
        path = self.output_dir / filename
        fieldnames = list(records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                # Flatten nested dicts
                flat = {}
                for k, v in record.items():
                    if isinstance(v, (list, dict)):
                        flat[k] = json.dumps(v)
                    elif isinstance(v, (np.integer, np.floating)):
                        flat[k] = float(v)
                    else:
                        flat[k] = v
                writer.writerow(flat)
        logger.info(f"CSV saved to {path}")
        return str(path)

    # ------------------------------------------------------------------
    # LaTeX table generation
    # ------------------------------------------------------------------
    def latex_table_depth_accuracy(
        self,
        results: List[ExperimentResult],
    ) -> str:
        """Generate LaTeX table for Experiment 1."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Experiment 1: Depth--Accuracy Tradeoff}",
            r"\label{tab:exp1_depth_accuracy}",
            r"\begin{tabular}{@{}ccccc@{}}",
            r"\toprule",
            r"Depth $L$ & F1 Score & Std.~Dev. & Precision Retained & Latency (ms) \\",
            r"\midrule",
        ]
        for r in results:
            lines.append(
                f"{r.depth} & {r.metric_value:.3f} & {r.metric_std:.3f} & "
                f"{r.precision_retained:.3f} & {r.latency_ms:.0f} \\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def latex_table_front_loading(
        self,
        results: Dict[str, List[ExperimentResult]],
    ) -> str:
        """Generate LaTeX table for Experiment 2."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Experiment 2: Budget Allocation Strategies ($L=3$)}",
            r"\label{tab:exp2_front_loading}",
            r"\begin{tabular}{@{}lcccc@{}}",
            r"\toprule",
            r"Strategy & Mean F1 & Std.~Dev. & Token Allocation & Rank \\",
            r"\midrule",
        ]

        means = {}
        for strategy, res_list in results.items():
            vals = [r.metric_value for r in res_list]
            means[strategy] = np.mean(vals) if vals else 0.0
        ranking = {
            s: i + 1 for i, (s, _) in enumerate(
                sorted(means.items(), key=lambda x: -x[1])
            )
        }

        for strategy, res_list in results.items():
            r = res_list[0]
            allocations = r.metadata.get("allocations", [])
            alloc_str = str(allocations).replace("[", "[").replace("]", "]")
            lines.append(
                f"{strategy.replace('_', '-')} & {r.metric_value:.3f} & "
                f"{r.metric_std:.3f} & {alloc_str} & {ranking[strategy]} \\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def latex_table_eta(
        self,
        estimates: List[DepreciationEstimate],
    ) -> str:
        """Generate LaTeX table for Experiment 3."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Experiment 3: Per-Layer Information Retention}",
            r"\label{tab:exp3_eta}",
            r"\begin{tabular}{@{}cccc@{}}",
            r"\toprule",
            r"Layer $\ell$ & Retention Rate & 95\% CI & Fitted $\hat\\eta^\\ell$ \\",
            r"\midrule",
        ]
        for e in estimates:
            lines.append(
                f"{e.layer} & {e.retention_rate:.3f} & "
                f"[{e.ci_lower:.3f}, {e.ci_upper:.3f}] & "
                f"{e.theoretical:.3f} \\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def export_all_latex(
        self,
        exp1_results: List[ExperimentResult],
        exp2_results: Dict[str, List[ExperimentResult]],
        exp3_estimates: List[DepreciationEstimate],
        filename: str = "tables.tex",
    ) -> str:
        """Export all LaTeX tables to a single file."""
        path = self.output_dir / filename
        content = (
            r"% Auto-generated LaTeX tables for Information Depreciation Experiments"
            + "\n% Generated: "
            + datetime.now().isoformat()
            + "\n\n"
        )
        content += self.latex_table_depth_accuracy(exp1_results)
        content += "\n\n\\vspace{1em}\n\n"
        content += self.latex_table_front_loading(exp2_results)
        content += "\n\n\\vspace{1em}\n\n"
        content += self.latex_table_eta(exp3_estimates)

        with open(path, "w") as f:
            f.write(content)
        logger.info(f"LaTeX tables saved to {path}")
        return str(path)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    def generate_summary(
        self,
        exp1: List[ExperimentResult],
        exp2: Dict[str, List[ExperimentResult]],
        exp3: List[DepreciationEstimate],
        eta_fitted: float,
        r_squared: float,
    ) -> str:
        """Generate a text summary of all experiments."""
        lines = []
        lines.append("=" * 70)
        lines.append("EXPERIMENT SUMMARY")
        lines.append("=" * 70)

        # Exp 1
        lines.append("\n--- Experiment 1: Depth-Accuracy Tradeoff ---")
        for r in exp1:
            lines.append(
                f"  L={r.depth}: F1={r.metric_value:.3f} (±{r.metric_std:.3f}), "
                f"Precision={r.precision_retained:.3f}"
            )
        f1s = [r.metric_value for r in exp1]
        pattern = "hump_shaped" if len(f1s) > 2 and f1s.index(max(f1s)) not in [0, len(f1s)-1] else "monotonic"
        lines.append(f"  Pattern: {pattern}")

        # Exp 2
        lines.append("\n--- Experiment 2: Front-Loading ---")
        means = {s: np.mean([r.metric_value for r in res]) for s, res in exp2.items()}
        for s, m in sorted(means.items(), key=lambda x: -x[1]):
            lines.append(f"  {s}: F1={m:.3f}")
        pred_ok = means.get("front_loaded", 0) > means.get("uniform", -1) > means.get("back_loaded", -2)
        lines.append(f"  Prediction (front > uniform > back): {pred_ok}")

        # Exp 3
        lines.append("\n--- Experiment 3: Eta Estimation ---")
        for e in exp3:
            lines.append(
                f"  Layer {e.layer}: Retention={e.retention_rate:.3f} "
                f"[{e.ci_lower:.3f}, {e.ci_upper:.3f}]"
            )
        lines.append(f"  Fitted eta = {eta_fitted:.3f}")
        lines.append(f"  R-squared = {r_squared:.3f}")

        return "\n".join(lines)


# =============================================================================
# Utility functions
# =============================================================================

def compute_f1(prediction: str, references: List[str]) -> float:
    """
    Compute token-level F1 score between prediction and references.

    Args:
        prediction: Predicted answer string.
        references: List of reference answer strings.

    Returns:
        F1 score in [0, 1].
    """
    def _normalize(text: str) -> str:
        """Normalize text: lowercase, remove articles/punctuation."""
        text = text.lower().strip()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"[^a-z0-9]", " ", text)
        return " ".join(text.split())

    def _f1(pred: str, ref: str) -> float:
        pred_tokens = _normalize(pred).split()
        ref_tokens = _normalize(ref).split()
        if not pred_tokens or not ref_tokens:
            return float(pred_tokens == ref_tokens)

        common = sum(
            (min(pred_tokens.count(t), ref_tokens.count(t)) for t in set(pred_tokens))
        )
        if common == 0:
            return 0.0
        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    if not references:
        return 0.0

    # Take maximum F1 against any reference
    return max(_f1(prediction, ref) for ref in references)


def fit_exponential_decay(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """
    Fit exponential decay: y = eta^x.

    Args:
        x: Independent variable (layer indices).
        y: Dependent variable (retention rates).

    Returns:
        (eta_estimate, r_squared).
    """
    # Log-transform: log(y) = x * log(eta)
    log_y = np.log(np.maximum(y, 0.001))

    if HAS_SCIPY:
        try:
            def model(x_data, log_eta):
                return x_data * log_eta
            popt, _ = curve_fit(model, x, log_y)
            eta_est = np.exp(popt[0])
        except Exception:
            # Fallback to linear regression
            slope = np.sum(x * log_y) / np.sum(x ** 2)
            eta_est = np.exp(slope)
    else:
        slope = np.sum(x * log_y) / np.sum(x ** 2)
        eta_est = np.exp(slope)

    # R-squared
    y_pred = eta_est ** x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(eta_est), float(r_squared)


def print_system_info() -> None:
    """Print system and GPU information."""
    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"    GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    logger.info(f"  transformers: {transformers.__version__ if HAS_TRANSFORMERS else 'N/A'}")
    logger.info(f"  datasets: {datasets.__version__ if HAS_DATASETS else 'N/A'}")
    logger.info(f"  vLLM: {'available' if HAS_VLLM else 'N/A'}")
    logger.info(f"  bitsandbytes: {'available' if HAS_BITSANDBYTES else 'N/A'}")
    logger.info(f"  scipy: {'available' if HAS_SCIPY else 'N/A'}")


# =============================================================================
# CLI Argument Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Run LLM experiments for 'Information Depreciation and Optimal Depth in AI Delegation Chains'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments with 7B model
  python run_real_experiments.py --experiment all --model_size 7b

  # Run only depth-accuracy experiment with vLLM
  python run_real_experiments.py --experiment depth --use_vllm --model_size 7b

  # Use 4-bit quantization for larger model
  python run_real_experiments.py --experiment all --model_size 13b --quantization 4bit

  # Resume interrupted experiment
  python run_real_experiments.py --experiment all --model_size 7b --resume

  # Estimate eta with custom parameters
  python run_real_experiments.py --experiment eta --model_size 7b --n_facts 200 --max_depth 5
        """,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["all", "depth", "front", "eta"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        default="7b",
        help="Model size to use (default: 7b)",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for accelerated inference",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Quantization mode (requires bitsandbytes)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum samples per experiment (default: 200)",
    )
    parser.add_argument(
        "--n_facts",
        type=int,
        default=100,
        help="Number of needle facts for eta estimation (default: 100)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum chain depth (default: 5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    return parser


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Main entry point for running experiments."""
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Print system info
    print_system_info()

    # Initialize model manager
    logger.info(f"\nInitializing ModelManager: {args.model_size}")
    model_manager = ModelManager(
        model_size=args.model_size,
        use_vllm=args.use_vllm,
        quantization=args.quantization,
        cache_dir=args.cache_dir,
        fallback=True,
    )

    # Initialize runner
    runner = ExperimentRunner(
        model_manager=model_manager,
        output_dir=args.output_dir,
    )

    exp1_results: List[ExperimentResult] = []
    exp2_results: Dict[str, List[ExperimentResult]] = {}
    exp3_estimates: List[DepreciationEstimate] = []
    eta_fitted = 0.92
    r_squared = 0.0

    # ======================================================================
    # Run experiments
    # ======================================================================
    try:
        # --- Experiment 1: Depth-Accuracy ---
        if args.experiment in ("all", "depth"):
            exp1_results = runner.run_depth_accuracy(
                depths=list(range(1, args.max_depth + 1)),
                max_samples=args.max_samples,
                resume=args.resume,
            )

        # --- Experiment 2: Front-Loading ---
        if args.experiment in ("all", "front"):
            exp2_results = runner.run_front_loading(
                depth=3,
                max_samples=args.max_samples,
                resume=args.resume,
            )

        # --- Experiment 3: Eta Estimation ---
        if args.experiment in ("all", "eta"):
            exp3_estimates = runner.run_eta_estimation(
                n_facts=args.n_facts,
                max_depth=args.max_depth,
                resume=args.resume,
            )
            if exp3_estimates:
                layers = np.arange(1, len(exp3_estimates) + 1)
                rates = np.array([e.retention_rate for e in exp3_estimates])
                eta_fitted, r_squared = fit_exponential_decay(layers, rates)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Saving partial results...")
    except Exception as e:
        logger.error(f"\nExperiment failed: {e}", exc_info=True)
        raise
    finally:
        # Unload model to free memory
        model_manager.unload()

    # ======================================================================
    # Analyze and export results
    # ======================================================================
    analyzer = ResultsAnalyzer(output_dir=args.output_dir)

    # Export LaTeX tables
    if exp1_results and exp2_results and exp3_estimates:
        analyzer.export_all_latex(exp1_results, exp2_results, exp3_estimates)

    # Export individual tables
    if exp1_results:
        latex = analyzer.latex_table_depth_accuracy(exp1_results)
        path = Path(args.output_dir) / "table_depth_accuracy.tex"
        with open(path, "w") as f:
            f.write(latex)
        logger.info(f"Depth-accuracy LaTeX table saved to {path}")

    if exp2_results:
        latex = analyzer.latex_table_front_loading(exp2_results)
        path = Path(args.output_dir) / "table_front_loading.tex"
        with open(path, "w") as f:
            f.write(latex)
        logger.info(f"Front-loading LaTeX table saved to {path}")

    if exp3_estimates:
        latex = analyzer.latex_table_eta(exp3_estimates)
        path = Path(args.output_dir) / "table_eta.tex"
        with open(path, "w") as f:
            f.write(latex)
        logger.info(f"Eta estimation LaTeX table saved to {path}")

    # Summary report
    if any([exp1_results, exp2_results, exp3_estimates]):
        summary = analyzer.generate_summary(
            exp1_results, exp2_results, exp3_estimates, eta_fitted, r_squared
        )
        summary_path = Path(args.output_dir) / "summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
        logger.info(f"\n{summary}")
        logger.info(f"\nSummary saved to {summary_path}")

    # Export JSON
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_size": args.model_size,
            "use_vllm": args.use_vllm,
            "quantization": args.quantization,
            "max_samples": args.max_samples,
            "seed": args.seed,
        },
        "experiment_1": [asdict(r) for r in exp1_results] if exp1_results else [],
        "experiment_2": {
            s: [asdict(r) for r in res] for s, res in exp2_results.items()
        } if exp2_results else {},
        "experiment_3": [asdict(e) for e in exp3_estimates] if exp3_estimates else [],
        "eta_fitted": eta_fitted,
        "r_squared": r_squared,
    }
    analyzer.export_json(all_results, "results.json")

    logger.info("\n" + "=" * 60)
    logger.info("All experiments completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
