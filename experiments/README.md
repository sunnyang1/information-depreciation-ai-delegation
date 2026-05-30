# LLM推理实验环境

本目录包含"信息折旧与AI代理链最优深度"论文的全部实验代码。

---

## 目录结构

```
experiments/
├── exp_framework.py          # 核心理论仿真引擎（公式驱动）
├── run.py                    # 统一实验运行器（命令行入口）
├── registry.py               # 实验注册表（自动发现26个实验）
├── run_real_experiments.py   # 真实LLM推理运行器（生产级）
├── exps/                     # 26个模块化实验（exp01–exp19 + sup01–sup07）
│   ├── exp01_depth_accuracy.py
│   ├── exp02_front_loading.py
│   ├── ...
│   └── sup07_sensitivity_analysis.py
├── tests/                    # 测试套件（44个pytest测试）
│   ├── test_exp_framework.py
│   └── test_integration.py
├── requirements.txt          # Python依赖
├── requirements-dev.txt      # 开发依赖（lint/test工具）
├── setup_env.sh              # AutoDL一键配置脚本
├── pyproject.toml            # 质量工具配置（black/isort/mypy/pylint/pytest）
├── Makefile                  # 常用命令封装
├── CODE_QUALITY.md           # 质量仪表盘
└── README.md                 # 本文件
```

---

## 快速开始

### 1. 统一实验运行器（推荐）

```bash
cd experiments

# 列出所有实验
python run.py --list

# 运行单个实验
python run.py --experiment exp01

# 运行全部26个实验
python run.py --experiment all

# 按类别运行（基础/高级/补充等）
python run.py --category v6_microfoundation

# 排除特定实验
python run.py --experiment all --exclude exp09,exp12
```

### 2. 模拟模式（无需GPU，理论验证）

```bash
cd experiments

# 直接运行核心引擎（3个基线实验）
python exp_framework.py

# 运行统一实验框架中的单个实验
python run.py --experiment exp01
```

### 3. 真实LLM推理

```bash
# 安装依赖
pip install -r requirements.txt

# 使用Llama-2-7B运行全部实验
python run_real_experiments.py --experiment all --model_size 7b

# 4-bit量化（显存不足时）
python run_real_experiments.py --experiment all --model_size 13b --quantization 4bit

# 从检查点恢复
python run_real_experiments.py --experiment all --model_size 7b --resume
```

---

## 实验清单

### 基础实验（v5 论文核心）

| 实验 | ID | 预测 | 运行命令 |
|------|-----|------|----------|
| 深度-精度权衡 | `exp01` | 精度随深度凹形下降 | `python run.py -e exp01` |
| 前载优势 | `exp02` | 前载 > 均匀 > 后载 | `python run.py -e exp02` |
| 指数衰减估计 | `exp03` | 保留率 ~ η^ℓ | `python run.py -e exp03` |
| 信号过载 | `exp04` | K↑ 则单信号精度↓ | `python run.py -e exp04` |
| 异质性减少失真 | `exp05` | 异质性越高 Δ 越低 | `python run.py -e exp05` |
| 成本无关性 | `exp06` | κ→0 时 L* 有限 | `python run.py -e exp06` |
| 预算扩张增加深度 | `exp07` | L* 随 A 增加 | `python run.py -e exp07` |

### 高级实验（v6 扩展）

| 实验 | ID | 预测 | 类别 |
|------|-----|------|------|
| 任务复杂度 | `exp08` | 任务越难，最优深度越浅 | `v5_extension` |
| 异构智能体类型 | `exp09` | Transformer前置最优 | `v5_extension` |
| 人机混合 | `exp10` | 人类-AI混合链存在最优深度 | `v5_extension` |
| IV模拟 | `exp11` | 工具变量识别折旧率 | `v5_extension` |
| 实验室协议 | `exp12` | 三种架构的参与者实验 | `v5_extension` |
| 率失真 | `exp13` | 逆向水填与并行高斯源 | `v6_microfoundation` |
| 记忆容量 | `exp14` | 记忆提升保留率但有上限 | `v6_microfoundation` |
| 跳跃连接 | `exp15` | 跳跃连接提高精度但深度仍有限 | `v6_architecture` |
| 并行分支 | `exp16` | 并行分支改善精度但仍衰减 | `v6_architecture` |
| 记忆增强链 | `exp17` | MACLA风格：更高保留，仍有限深度 | `v6_architecture` |
| 后载边界 | `exp18` | 凸后期收益偏好后载 | `v6_frontloading` |
| 前载充分条件 | `exp19` | 凹价值下前载占优 | `v6_frontloading` |

### 补充实验

| 实验 | ID | 说明 |
|------|-----|------|
| 精度衰减曲线 | `sup01` | 不同 η̄ 下的保留率曲线 |
| ρ vs 预算 | `sup02` | 传输因子随预算变化 |
| 利润函数 | `sup03` | 利润随深度变化 |
| 最优深度 vs 预算 | `sup04` | L* 随上下文窗口增加 |
| 保留率与边际损失 | `sup05` | 累积保留与边际损失 |
| GHM基准 | `sup06` | 与GHM模型对比 |
| 敏感性分析 | `sup07` | γ 和最优深度网格敏感性 |

---

## 代码质量

本项目已配置完整的自动化质量保障体系。

### 一键运行全部检查

```bash
cd experiments
make ci      # lint + typecheck + smoke + unit-test
```

### 单独运行

```bash
# 代码风格
flake8 --max-line-length=100 --ignore=E203,W503,E741,E731,E402 \
  exp_framework.py run.py registry.py run_real_experiments.py

# 类型检查
mypy exp_framework.py run.py registry.py --ignore-missing-imports

# 单元测试 + 覆盖率
pytest ../tests/ -v --cov=exp_framework --cov-report=term-missing

# 安全扫描
bandit -r . -ll
```

### 开发环境

```bash
pip install -r requirements-dev.txt
pre-commit install
```

---

## AutoDL部署

### 一键配置

```bash
bash setup_env.sh
```

脚本会自动完成：
1. 创建conda环境 `info_depreciation`
2. 安装匹配CUDA版本的PyTorch
3. 安装Transformers、Datasets、vLLM等依赖
4. 配置HuggingFace国内镜像
5. 创建实验目录结构

### 手动配置（备选）

```bash
conda create -n info_depreciation python=3.10 -y
conda activate info_depreciation
pip install -r requirements.txt
```

### 国内镜像配置

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface_cache
```

---

## 常见问题

**Q: 模型下载失败？**
> 检查HF镜像：`echo $HF_ENDPOINT` 应输出 `https://hf-mirror.com`

**Q: CUDA版本不匹配？**
> `setup_env.sh` 会自动检测CUDA版本。如仍有问题，手动安装对应版本PyTorch。

**Q: 显存不足？**
> 使用 `--quantization 4bit` 或 `--quantization 8bit`。

**Q: 模拟实验和真实实验的区别？**
> 模拟实验使用理论忠实的数学引擎（无需GPU/模型下载），验证结构预测。真实实验使用HuggingFace模型进行实际推理，验证实证预测。

**Q: 如何添加新实验？**
> 在 `exps/` 下创建新文件，使用 `@register()` 装饰器注册。运行 `python run.py --list` 即可看到新实验。

---

## 引用

如果您在研究中使用了本实验代码，请引用：

```bibtex
@article{info_depreciation_2026,
  title={Information Depreciation and Optimal Depth in AI Delegation Chains},
  author={[Authors Redacted]},
  year={2026},
  note={Working paper}
}
```
