# LLM推理实验环境

本目录包含"信息折旧与AI代理链最优深度"论文的全部实验代码。

---

## 目录结构

```
experiments/
├── exp_framework.py          # 基线模拟实验（实验1-3）
├── exp_advanced.py           # 高级模拟实验（实验4-7）
├── run_real_experiments.py   # 真实LLM推理运行器
├── visualize_results.py      # 结果可视化
├── requirements.txt          # Python依赖
├── setup_env.sh              # AutoDL一键配置脚本
└── README.md                 # 本文件
```

---

## 快速开始

### 模拟模式（无需GPU）

```bash
cd experiments
python exp_framework.py       # 运行基线实验1-3
python exp_advanced.py        # 运行高级实验4-7
python visualize_results.py   # 生成图表
```

结果保存在 `experiments/results/` 和 `experiments/figures/`。

### 真实LLM推理

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

## 实验说明

### 实验1：深度-精度权衡
- **设计**：固定总预算，均匀分配，变化链深度 L=1..5
- **预测**：精度随深度呈凹形下降（Prediction 7.1）
- **运行**：`python exp_framework.py`（自动包含）

### 实验2：前载优势
- **设计**：固定总预算和深度 L=3，对比均匀/前载/后载分配策略
- **预测**：前载 > 均匀 > 后载（Proposition 4）
- **运行**：`python exp_framework.py`（自动包含）

### 实验3：指数衰减估计
- **设计**：在层0插入100个单位精度信号，测量每层保留率
- **预测**：保留率呈指数衰减 ~ η^ℓ（Proposition 5）
- **运行**：`python exp_framework.py`（自动包含）

### 实验4：信号过载
- **设计**：固定预算 A=30K tokens，变化信号数量 K=20..500
- **预测**：固定预算下，K 增加会降低单信号精度（Prediction 7.2）
- **运行**：`python exp_advanced.py`（自动包含）

### 实验5：异质性减少失真
- **设计**：控制总精度相同，对比同质 vs 异质信号分布
- **预测**：异质性越高，聚合失真 Δ 越低（Prediction 7.3）
- **运行**：`python exp_advanced.py`（自动包含）

### 实验6：成本无关性
- **设计**：变化成本参数 κ，计算最优深度 L*
- **预测**：κ→0 时 L* 趋于有限值（Prediction 7.4）
- **运行**：`python exp_advanced.py`（自动包含）

### 实验7：预算扩张增加深度
- **设计**：变化上下文窗口大小 A，计算最优深度 L*
- **预测**：L* 随 A 增加而增加（Prediction 7.5）
- **运行**：`python exp_advanced.py`（自动包含）

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
