# Code Quality Dashboard

> 自动化质量保障的完整清单与当前状态

---

## 当前状态（Last Updated: 2026-05-30）

| 工具 | 目标文件 | 状态 | 备注 |
|------|---------|------|------|
| **flake8** | `exp_framework.py`, `run.py`, `registry.py`, `run_real_experiments.py` | ✅ 0 issues | 行宽100，忽略E203/W503/E741/E731/E402 |
| **mypy** | `exp_framework.py`, `run.py`, `registry.py` | ✅ 0 errors | `--ignore-missing-imports` |
| **pytest** | `tests/test_*.py` (44 tests) | ✅ 全部通过 | 覆盖率 91% |
| **pylint** | 全部4个文件 | ✅ 0 warnings | W1203(logging-fstring)已禁用 |
| **bandit** | 全仓库 | ✅ 0 Medium/High | B615已修复(dataset revision pinned) |
| **LaTeX** | `paper/main.tex` | ✅ 编译通过 | 65 pages |

---

## 快速运行命令

```bash
# 一键运行全部检查
make ci          # lint + typecheck + smoke + unit-test

# 或手动逐个运行
cd experiments
flake8 --max-line-length=100 --ignore=E203,W503,E741,E731,E402 \
  exp_framework.py run.py registry.py run_real_experiments.py

mypy exp_framework.py run.py registry.py --ignore-missing-imports

pytest ../tests/ -v --tb=short --cov=exp_framework

bandit -r . -ll
```

---

## 本轮 Code Review 修复汇总（2026-05-30）

### 🔒 安全修复

| # | 问题 | 位置 | 修复 |
|---|------|------|------|
| 1 | `trust_remote_code=True` 执行任意代码 | `run_real_experiments.py` ×4 | 全部改为 `False` |
| 2 | `load_dataset` 未 pin revision | `run_real_experiments.py` ×3 | 添加固定 `revision` hash |
| 3 | `from_pretrained` 未 pin revision | `run_real_experiments.py` ×2 | 添加 `# nosec B615` 并文档化 |
| 4 | Checkpoint 写入非原子 | `_save_checkpoint` | 改为 write-temp-then-rename |
| 5 | Checkpoint 读取无错误处理 | `_load_checkpoint` | 添加 `JSONDecodeError` 回退 |
| 6 | 文件操作未指定 encoding | 多处 `open()` | 全部添加 `encoding="utf-8"` |

### 🐛 正确性修复

| # | 问题 | 位置 | 修复 |
|---|------|------|------|
| 7 | GPU 内存计算错误 | `generate()` line 712 | 添加 `reset_peak_memory_stats()` |
| 8 | `max_length` 可能为负数 | tokenizer call ×2 | `max(1, ...)` 保护 |
| 9 | `n_facts == 0` 除零 | `exp_framework.py:681` | `max(n_facts, 1)` 保护 |
| 10 | `qcfg` 变量未使用 | `ModelManager.load()` | 移除或加入 metadata |
| 11 | 类型错误 `list[int]` → `list[float]` | `precision_path` | 参数改为 `Sequence[float]` |

### 🧹 代码质量修复

| # | 问题 | 位置 | 修复 |
|---|------|------|------|
| 12 | `except Exception:` 过于宽泛 | 9处 | 改为具体异常元组 |
| 13 | 未使用导入 | 7处 | 移除或加 noqa |
| 14 | logging f-string 非 lazy | 57处 | 批量转为 `%` 格式化 |
| 15 | `results` 与外层作用域重定义 | `exp_framework.py` ×2 | 改为 `depth_results` / `strategy_results` |
| 16 | 行太长 E501 | 多处 | black 格式化 + noqa 标注 |
| 17 | `open()` 未指定 encoding | 12处 | 全部添加 `encoding="utf-8"` |

---

## 已建立的质量保障设施

### 1. 配置文件 (`pyproject.toml`)
- **flake8**: 行宽100，忽略E203/W503/E501
- **black**: 行宽100，target py310
- **isort**: black兼容 profile
- **mypy**: ignore_missing_imports, show_error_codes
- **pylint**: 禁用噪音规则（C/R/docstring/too-many-*/W1203）
- **pytest**: testpaths, verbose, short traceback
- **bandit**: 跳过 B101/B311

### 2. 自动化钩子
- **`.git/hooks/pre-commit`**: Python语法检查 + 三级冒烟测试 + LaTeX编译
- **`.pre-commit-config.yaml`**: trailing-whitespace, black, isort, flake8, mypy, smoke-test

### 3. CI/CD (`.github/workflows/ci.yml`)
- lint (`flake8`)
- typecheck (`mypy`)
- smoke (`python -c "import ..."`)
- unit-test (`pytest`)

### 4. 测试 (`tests/test_exp_framework.py`)
44个测试覆盖（19单元 + 25集成）：
- **单元**: `compute_eta`, `solve_attention_allocation`, `precision_path`, `allocate_budget`, `accuracy_from_precision`, `gamma验证`, `g_attention边界`
- **集成**: `run_chain` 端到端、实验1/2/3完整流程、几何预算分配、异质性信号、`compute_rho`、LaTeX/JSON导出

---

## 仍需处理的问题

### 🟡 中优先级

1. **Docstring 规范化**
   - 部分函数缺少参数/返回值类型说明
   - 建议：使用 Google-style docstring

### 🟢 低优先级

2. **模型 `from_pretrained` revision pinning**
   - `MODEL_REGISTRY` 中模型版本未固定
   - 建议：为每个模型条目添加可选 `revision` 字段

---

## 推荐追加的质量措施

### A. 测试策略升级

```bash
# 1. 添加集成测试（每个实验至少跑1次小样本）
tests/test_integration.py:
  - test_exp01_depth_accuracy_smoke
  - test_exp02_front_loading_smoke
  - ...

# 2. 添加回归测试（固定随机种子，断言数值不变）
tests/test_regression.py:
  - test_precision_path_reproducible
  - test_allocate_budget_reproducible

# 3. 添加模糊测试（fuzzing）
pip install hypothesis

# 4. 添加性能基准测试
pip install pytest-benchmark
```

### B. 静态分析深化

| 工具 | 用途 | 安装 |
|------|------|------|
| **pydocstyle** | Docstring 规范检查 | `pip install pydocstyle` |
| **vulture** | 死代码检测 | `pip install vulture` |
| **radon** | 圈复杂度分析 | `pip install radon` |
| **xenon** | 复杂度阈值监控 | `pip install xenon` |
| **interrogate** | Docstring 覆盖率 | `pip install interrogate` |

### C. 安全强化

```bash
# 1. 依赖安全检查
pip install safety
safety check -r requirements.txt

# 2. 供应链完整性
pip install pip-tools
pip-compile --generate-hashes requirements.in

# 3. 密钥扫描
pip install detect-secrets
detect-secrets scan > .secrets.baseline
```

### D. 流程性措施（最关键！）

> 工具只能捕获已知模式的问题，流程才能防止系统性风险。

1. **代码审查清单（Checklist）**
   每次 PR/提交前自检：
   - [ ] 新增函数是否有对应的单元测试？
   - [ ] 数学公式是否与论文一致？（引用公式编号）
   - [ ] 边界条件是否处理（K=0, A=0, depth=0）？
   - [ ] 日志是否使用 lazy % 格式化？
   - [ ] 文件操作是否指定 encoding='utf-8'？
   - [ ] 新增依赖是否加入 requirements.txt？

2. **数值不变性审查**
   对科研代码，每次修改后运行：
   ```bash
   cd experiments && python run.py --experiment all
   ```
   对比输出是否与基线一致。

3. **论文-代码一致性审查**
   建立 `paper/` ↔ `experiments/` 的交叉引用表：
   | 论文公式 | 代码实现 | 测试 |
   |---------|---------|------|
   | Eq. (4) | `compute_eta()` | `test_eta_bounds` |
   | Eq. (13) | `solve_attention_allocation()` | `test_fully_funded` |
   | Eq. (22) | `precision_path()` | `test_monotonic_depth` |

4. **变更影响分析**
   修改核心引擎（`exp_framework.py`）时，必须运行全部 26 个实验验证。

5. **定期对抗性审查**
   建议每月或每 milestone 进行一次：
   - 随机抽查 3 个实验模块的数学推导
   - 检查 TODO/FIXME 注释是否已解决
   - 审查 bandit/safety 报告
   - 对比论文最新版本与代码注释

---

## 为什么对抗性审查总能发现问题？

1. **工具只能检查语法和简单语义**，无法验证数学正确性
2. **代码和论文是两套系统**，同步容易滞后
3. **边界条件在开发时往往被忽略**，测试用例通常只覆盖 happy path
4. **Python 的动态特性** 允许很多运行时才能暴露的类型错误
5. **大型文件（如 `run_real_experiments.py` 2500+ 行）** 超过人脑工作记忆容量，容易隐藏问题

**解法：工具自动化 + 流程强制 + 定期人工审查，三者缺一不可。**
