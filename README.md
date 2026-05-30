# Information Depreciation and Optimal Depth in AI Delegation Chains

[![Python 3.10+](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Paper**: "Information Depreciation and Optimal Depth in AI Delegation Chains"  
> **Authors**: [Redacted for Peer Review]  
> **Date**: May 2026

## Overview

This repository contains the source code and LaTeX paper for a research project that embeds **rational inattention** into multi-layer AI delegation chains. The core question is: *when AI agents delegate to other AI agents, how does information depreciate at each processing layer, and what bounds the optimal organizational depth?*

### Three Core Contributions

1. **Information Depreciation Theory**: We model signal precision as a depreciating asset that traverses a processing chain, with an *endogenous* preservation rate determined by attention allocation.
2. **Front-Loading Proposition**: The optimal budget allocation is front-loaded—larger context windows at early layers maximize output quality (Proposition 4).
3. **Finite Optimal Depth**: Even with costless communication, endogenous constraints on information transmission bound the efficient depth of the chain (Proposition 6).

### Seven Testable Predictions

The experiments validate seven theoretical predictions:

| # | Prediction | Paper Reference | Experiment |
|---|-----------|-----------------|------------|
| 1 | Depth-Accuracy Tradeoff (concave) | Prediction 7.1 | `exp01` |
| 2 | Front-Loading Advantage | Proposition 4 | `exp02` |
| 3 | Exponential Decay of Retention | Proposition 5 | `exp03` |
| 4 | Signal Overload | Prediction 7.2 | `exp04` |
| 5 | Heterogeneity Reduces Distortion | Prediction 7.3 | `exp05` |
| 6 | Cost Irrelevance for Depth | Prediction 7.4 | `exp06` |
| 7 | Budget Expansion Increases Depth | Prediction 7.5 | `exp07` |

---

## Repository Structure

```
.
├── paper/                          # LaTeX paper source
│   ├── main.tex                    # Root document
│   ├── sec_intro.tex               # Introduction
│   ├── sec_model.tex               # Theoretical model
│   ├── sec_single.tex              # Single-layer analysis
│   ├── sec_multi.tex               # Multi-layer analysis
│   ├── sec_calibration.tex         # Model calibration
│   ├── sec_empirical.tex           # Empirical predictions
│   ├── sec_experiments.tex         # Experimental validation
│   └── ...                         # Other sections
├── experiments/                    # Python experiment code
│   ├── exp_framework.py            # Core simulation engine (3 baseline experiments)
│   ├── run.py                      # Unified runner for all registered experiments
│   ├── registry.py                 # Experiment registry
│   ├── exps/                       # 19 modular experiment definitions
│   ├── run_real_experiments.py     # Production real-LLM inference runner
│   ├── requirements.txt            # Python dependencies
│   ├── setup_env.sh                # AutoDL one-click environment setup
│   └── README.md                   # Experiment guide (Chinese)
├── simulation_regression.py        # Simulated regression analysis + LaTeX tables
├── EXPERIMENTS.md                  # Detailed experiment design document
├── AGENTS.md                       # Agent-facing development guide
└── README.md                       # This file
```

---

## Quick Start

### Simulation Mode (No GPU Required)

Run all baseline experiments directly:

```bash
cd experiments
python exp_framework.py
```

Run the unified experiment runner (all 26 registered experiments):

```bash
cd experiments
python run.py --experiment all

# Or run by category
python run.py --category baseline
python run.py --category advanced
python run.py --category reviewer
python run.py --category supplementary

# List all experiments
python run.py --list
```

Results are saved to `experiments/results/` (JSON + LaTeX tables).

### Regression Analysis

Generate simulated data, run OLS regressions, and export LaTeX tables:

```bash
python simulation_regression.py
# Output: regression_tables.tex
```

### Real LLM Inference Mode

```bash
cd experiments
# Install dependencies
pip install -r requirements.txt

# Run with Llama-2-7B
python run_real_experiments.py --experiment all --model_size 7b

# 4-bit quantization for limited VRAM
python run_real_experiments.py --experiment all --model_size 13b --quantization 4bit

# Use Llama-3.1-8B with 128K context (recommended)
python run_real_experiments.py --experiment all --model_size llama3_8b
```

### LaTeX Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Key Parameters

| Parameter | Symbol | Illustrative Value | Source |
|-----------|--------|-------------------|--------|
| Attention elasticity | γ | 0.35 | MMLU long-context scores |
| Min preservation rate | η̲ | 0.75 | Needle-in-Haystack benchmarks |
| Max preservation rate | η̄ | 0.95 | Theoretical upper bound |
| Saturation parameter | a | 2.0 | Calibrated |
| Context window | A | 4K–1M tokens | API documentation |

See `paper/sec_calibration.tex` for full calibration details.

---

## Supported Models

| Key | Model | Context | Best For |
|-----|-------|---------|----------|
| `7b` | unsloth/llama-2-7b-chat | 4K | Fast testing |
| `llama3_8b` | meta-llama/Llama-3.1-8B-Instruct | 128K | **Recommended** |
| `mistral7b` | mistralai/Mistral-7B-Instruct-v0.3 | 32K | Public, fast |
| `qwen7b` | Qwen/Qwen2.5-7B-Instruct | 32K | Public, no gate |
| `13b` | meta-llama/Llama-2-13b-chat-hf | 4K | Medium quality |
| `70b` | meta-llama/Llama-2-70b-chat-hf | 4K | High quality (multi-GPU) |
| `tiny` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 2K | Smoke test |
| `phi2` | microsoft/phi-2 | 2K | Small, good quality |

---

## Citation

If you use this code or build on our framework, please cite:

```bibtex
@article{info_depreciation_2026,
  title={Information Depreciation and Optimal Depth in AI Delegation Chains},
  author={[Authors Redacted]},
  year={2026},
  note={Working paper}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
# test
