# AGENTS.md — Information Depreciation and Optimal Depth in AI Delegation Chains

> **For AI coding agents**: This file describes the project structure, build/run processes, and development conventions.

---

## Project Overview

This repository contains the source code and LaTeX paper for **"Information Depreciation and Optimal Depth in AI Delegation Chains."**

The project combines economic theory (rational inattention, organizational economics) with large language model (LLM) experiments. The core research question: when AI agents delegate to other AI agents, how does information depreciate at each processing layer, and what bounds the optimal organizational depth?

### Two Main Components

1. **Paper** (`paper/`): LaTeX academic paper with theorem environments, mathematical proofs, and empirical validation sections.
2. **Experiments** (`experiments/`): Python code validating theoretical predictions through simulation and real LLM inference.

### Seven Testable Predictions

The experiments validate seven theoretical predictions:
1. **Depth-Accuracy Tradeoff**: Accuracy degrades with chain depth (hump-shaped or monotonically decreasing).
2. **Front-Loading Advantage**: Front-loaded budget allocation outperforms uniform or back-loaded strategies.
3. **Exponential Decay**: Information retention follows exponential decay ~ η^ℓ across layers.
4. **Signal Overload**: For fixed context window, increasing K decreases per-signal precision.
5. **Heterogeneity Reduces Distortion**: Holding total precision constant, heterogeneous signal distributions yield lower aggregate distortion.
6. **Cost Irrelevance for Depth**: Reducing API costs does not increase optimal depth beyond a finite limit.
7. **Budget Expansion Increases Depth**: Larger context windows permit deeper optimal chains.

---

## Technology Stack

| Component | Version / Tool | Purpose |
|-----------|---------------|---------|
| Python | 3.10+ | Runtime environment |
| PyTorch | 2.7.0 | Deep learning framework |
| Transformers | >=4.40.0 | Model loading and inference |
| Datasets | 2.14.0 | HuggingFace dataset loading |
| vLLM | >=0.11.0 | Accelerated LLM inference (optional) |
| bitsandbytes | >=0.43.0 | 4-bit / 8-bit quantization |
| accelerate | >=0.24.0 | Multi-GPU support |
| NumPy / SciPy / pandas | latest compatible | Data processing and statistics |
| statsmodels | latest compatible | OLS regression analysis |
| matplotlib | latest compatible | Visualization |
| LaTeX | pdfTeX/XeLaTeX | Paper compilation |

**Platform target**: GPU cloud platforms (AutoDL, Together AI, or institutional clusters). Simulation mode requires no GPU.

---

## Directory Structure

```
.
├── paper/                          # LaTeX paper source
│   ├── main.tex                    # Root document
│   ├── sec_intro.tex               # Introduction
│   ├── sec_literature.tex          # Literature review
│   ├── sec_model.tex               # Theoretical model
│   ├── sec_single.tex              # Single-layer analysis
│   ├── sec_multi.tex               # Multi-layer analysis
│   ├── sec_calibration.tex         # Model calibration
│   ├── sec_empirical.tex           # Empirical predictions
│   ├── sec_experiments.tex         # Experimental validation
│   └── ...                         # Other sections
├── experiments/                    # Python experiment code
│   ├── exp_framework.py            # Baseline simulation engine (3 experiments)
│   ├── run.py                      # Unified experiment runner (registry-based)
│   ├── run_real_experiments.py     # Production real-LLM runner (~2400 lines)
│   ├── exps/                       # Modular experiment definitions (19 experiments)
│   ├── registry.py                 # Experiment registry
│   ├── requirements.txt            # Python dependencies
│   ├── setup_env.sh                # AutoDL one-click environment setup
│   └── README.md                   # Experiment guide (Chinese)
├── simulation_regression.py        # Simulated regression analysis
├── EXPERIMENTS.md                  # Detailed experiment design document
├── README.md                       # Project overview
└── AGENTS.md                       # This file
```

---

## Build and Run Commands

### LaTeX Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Python Experiments

#### Simulation Mode (No GPU Required)

```bash
cd experiments
python exp_framework.py          # Baseline experiments 1-3
python run.py --experiment all   # All registered simulation experiments
python run.py --category baseline # Baseline category only
```

Outputs:
- `results/results.json` — structured JSON with all metrics
- `results/unified_results_*.json` — unified runner output
- `results/*.tex` — LaTeX tables ready for inclusion in paper
- `figures/*.png` — publication-quality figures

#### Real LLM Inference Mode

```bash
# Setup (AutoDL)
bash experiments/setup_env.sh

# Run all experiments with Llama-2-7B
python run_real_experiments.py --experiment all --model_size 7b

# 4-bit quantization for limited VRAM
python run_real_experiments.py --experiment all --model_size 13b --quantization 4bit

# Resume from checkpoint
python run_real_experiments.py --experiment all --model_size 7b --resume
```

#### Regression Analysis

```bash
python simulation_regression.py
```

---

## Code Organization

### `experiments/exp_framework.py` — Theory-Faithful Simulation Engine

- **`TheoreticalLLMChain`**: Core engine implementing structural equations from the paper.
  - `compute_eta(A, K)`: Endogenous preservation rate (Assumption 2, Eq. 4)
  - `solve_attention_allocation(A, tau0_list)`: Optimal attention weights (Proposition 1, Eq. 13)
  - `compute_rho(A, tau0_list)`: Endogenous transmission factor (Definition 3, Eq. 16)
  - `precision_path(budgets, K_list)`: Recursive precision dynamics (Proposition 5, Eq. 22)
  - `accuracy_from_precision(tau)`: Calibrated sigmoid mapping to observable accuracy
- **`allocate_budget()`**: Budget allocation strategies (uniform, front-loaded, back-loaded, geometric)
- **Three baseline experiments**:
  - `run_experiment_1_depth_accuracy()`: Vary depth L = 1..5
  - `run_experiment_2_front_loading()`: Compare strategies at L = 3
  - `run_experiment_3_eta_estimation()`: Estimate per-layer retention rate

### `experiments/exps/` — Modular Experiment Definitions

- **Baseline**: `exp01_depth_accuracy`, `exp02_front_loading`, `exp03_exponential_decay`
- **Advanced**: `exp04_signal_overload`, `exp05_heterogeneity`, `exp06_cost_irrelevance`, `exp07_budget_depth`
- **Reviewer Response**: `exp08`–`exp12`
- **V6 Architecture**: `exp15`–`exp17`
- **V6 Front-Loading**: `exp18`–`exp19`
- **V6 Microfoundation**: `exp13`–`exp14`
- **Supplementary Figures**: `sup01`–`sup07`

### `experiments/run.py` — Unified Experiment Runner

- **Registry-based dispatch**: Discovers and executes all experiments registered via `@register` decorator
- **Category filtering**: Run subsets (`baseline`, `advanced`, `reviewer`, `supplementary`, `v6_*`)
- **Result export**: Saves JSON results and LaTeX stubs with per-experiment status and metadata

### `experiments/run_real_experiments.py` — Production Runner

- **`ModelManager`**: Model loading via Transformers/vLLM, quantization, multi-GPU
- **`DatasetLoader`**: SQuAD v2, HotpotQA, GSM8K, Needle-in-Haystack
- **`LLMChain`**: Multi-layer inference with output-to-input passing
- **`ExperimentRunner`**: Checkpoint/resume, result export (JSON/CSV/LaTeX)

### `simulation_regression.py` — Regression Analysis

- Generates simulated datasets for Needle-in-Haystack, Depth-Accuracy, and Signal-Overload scenarios
- Runs OLS regressions with robust standard errors (`HC1`)
- Exports formatted LaTeX regression tables to `regression_tables.tex`

---

## Key Configuration and Constants

### Model Registry (`run_real_experiments.py`)

Supported model sizes:
- `"7b"` → `unsloth/llama-2-7b-chat` (public mirror, no Meta gate)
- `"falcon7b"` → `tiiuae/falcon-7b-instruct` (fully public)
- `"mistral7b"` → `mistralai/Mistral-7B-Instruct-v0.3` (public, fast)
- `"13b"` → `NousResearch/Llama-2-13b-chat-hf` (public mirror, no HF gate)
- `"llama3_8b"` → `unsloth/Llama-3.1-8B-Instruct` (recommended, 128K context, public mirror)
- `"70b"` → `meta-llama/Llama-2-70b-chat-hf`
- `"tiny"` → `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `"qwen7b"` → `Qwen/Qwen2.5-7B-Instruct` (public, no HF gate)
- `"phi2"` → `microsoft/phi-2`

### Simulation Parameters (`exp_framework.py`)

- `ETA_UNDERLINE = 0.75` (min preservation rate)
- `ETA_BAR = 0.95` (max preservation rate)
- `A_SAT = 2.0` (saturation parameter)
- `GAMMA = 0.35` (attention elasticity)
- `KAPPA = 1e-5` (linear cost coefficient)
- `TOKENS_PER_ATTENTION_UNIT = 1000.0` (budget scaling)
- `DEFAULT_K = 100` (signals per layer)
- `DEFAULT_TOTAL_BUDGET = 200_000` (tokens)

---

## Development Conventions

### Code Style

- **Python**: PEP 8 broadly. Type hints, Google-style docstrings, snake_case.
- **Line length**: ~100 characters.
- **String formatting**: f-strings preferred.

### Error Handling

- Real inference code wraps dataset loading and model inference in `try/except`
- `ModelManager` has automatic fallback to smaller models
- Checkpoints saved after each sub-experiment

### Logging

- `run_real_experiments.py` uses `logging.basicConfig` with INFO level
- `exp_framework.py` and `exps/*` modules use `print()` for simulation output

### Data Output

- Results saved to `./experiments/results/` by default
- JSON for structured data, `.tex` for LaTeX tables, `.png` for figures

---

## Testing Strategies

There is **no formal test suite**. Testing is done through:

1. **Simulation mode**: Run `exp_framework.py` or `run.py --experiment all` to verify structural predictions
2. **Mock data mode**: `DatasetLoader` falls back to mock data if HF is unreachable
3. **Tiny model smoke test**: Run with `--model_size tiny` for fast real-inference check
4. **Manual verification**: Check generated JSON and figures for expected patterns

---

## Deployment Process

### AutoDL (Primary Target Platform)

1. Create instance with PyTorch 2.x + CUDA 11.8/12.1 image
2. Run `bash setup_env.sh`
3. Activate environment: `conda activate info_depreciation`

**Simulation mode (no GPU required):**
```bash
cd /root/autodl-tmp/experiments
python run.py --experiment all
```

**Real LLM inference mode:**
```bash
cd /root/autodl-tmp/experiments
python run_real_experiments.py --experiment all --model_size 7b
```

### Environment Variables for China Deployment

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface_cache
```

These are automatically set by `setup_env.sh`.

---

## Notes for Agents

- **Bilingual documentation**: Paper and code comments are in English; deployment guides (`README.md`, `autodl_guide.md`, `setup_env.sh` comments) are in Chinese. Maintain existing language of each file.
- **No package manager config**: Dependencies managed via `requirements.txt` and `setup_env.sh`.
- **LaTeX integration**: Experiment scripts generate `.tex` table files. These are manually copied into `paper/sec_experiments.tex` or included via `\input`. No automated build pipeline links experiments to the paper.
- **Large file caution**: Downloading models (e.g., Llama-2-70B) requires tens of GB of disk space. Use `--quantization 4bit` and model sizes `7b`, `llama3_8b`, or `tiny` unless you have multiple A100s.
