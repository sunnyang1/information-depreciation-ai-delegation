# AutoDL GPU云平台完整使用指南

> **适用对象**：经济学研究者和学术用户  
> **用途**：在AutoDL平台上运行LLM（大语言模型）实验  
> **版本**：2025年6月  
> **平台官网**：[https://www.autodl.com](https://www.autodl.com)

---

## 目录

- [第一部分：注册和准备](#第一部分注册和准备)
- [第二部分：实例选择](#第二部分实例选择)
- [第三部分：环境配置](#第三部分环境配置)
- [第四部分：运行实验](#第四部分运行实验)
- [第五部分：成本优化](#第五部分成本优化)
- [第六部分：代码示例](#第六部分代码示例)
- [第七部分：故障排除](#第七部分故障排除)

---

## 第一部分：注册和准备

### 1.1 注册AutoDL账号

**步骤：**

1. 打开浏览器，访问 AutoDL 官网：https://www.autodl.com
2. 点击页面右上角的 **"注册"** 按钮
3. 输入手机号码，获取验证码
4. 设置登录密码（建议包含大小写字母+数字）
5. 勾选用户协议，点击 **"注册"**

> **界面说明**：注册页面包含手机号输入框、验证码获取按钮、密码输入框和用户协议勾选框。

**【截图位置 - 图1-1】**：AutoDL官网首页，右上角显示"注册"和"登录"按钮

---

### 1.2 实名认证

AutoDL 要求完成实名认证才能创建实例。认证流程如下：

**步骤：**

1. 登录账号后，点击右上角头像 → **"账号设置"**
2. 左侧菜单选择 **"实名认证"**
3. 选择认证类型：
   - **个人认证**：上传身份证正反面照片，进行人脸识别
   - **学生认证**（如有.edu邮箱）：可享受额外优惠
4. 按照页面提示完成认证，通常 **10分钟内** 审核通过

> **提示**：学生认证通过后，部分GPU可享受学生折扣价格。

**【截图位置 - 图1-2】**：实名认证页面，显示身份证上传区域和人脸识别按钮

---

### 1.3 充值

AutoDL 支持多种充值方式，操作便捷：

**步骤：**

1. 登录后点击右上角头像 → **"充值中心"**
2. 选择充值金额，支持自定义输入
3. 选择支付方式：
   - **支付宝**（推荐）
   - **微信支付**
   - **银联卡**
4. 扫码或确认支付，余额实时到账

**【截图位置 - 图1-3】**：充值中心页面，显示金额选择框和支付方式选项

---

### 1.4 推荐起充金额

| 用户类型 | 推荐金额 | 可用时长估算 | 说明 |
|---------|---------|------------|------|
| 试用体验 | 50元 | 约50小时（RTX 3090） | 适合初次体验 |
| 经济型研究 | 100元 | 约100小时（RTX 3090） | 推荐起充金额 |
| 标准研究 | 200元 | 约130小时（RTX 4090） | 适合完整实验周期 |
| 大规模实验 | 500元 | 约100小时（A100） | 大模型训练需求 |

> **建议**：首次充值 **100-200元** 足够完成多个实验。AutoDL按秒计费，关机即停止扣费，余额可长期使用。

---

### 1.5 领取新用户优惠券

AutoDL 经常为新用户提供优惠券，领取方式：

**步骤：**

1. 注册并登录后，访问 **"控制台"** → **"优惠券"** 页面
2. 查看可用的优惠券列表
3. 点击 **"领取"** 按钮
4. 新用户通常可获得 **5-20元代金券**

**其他获取优惠的渠道：**

- 关注 AutoDL 官方微信公众号，不定期发放优惠券
- 参加 AutoDL 社区活动，可获得算力券奖励
- 学生认证通过后，自动发放学生专属优惠券

**【截图位置 - 图1-4】**：优惠券页面，显示可用优惠券列表和领取按钮

---

## 第二部分：实例选择

### 2.1 AutoDL GPU类型和价格

AutoDL 提供多种 NVIDIA GPU，以下是主要型号和价格参考（2025年6月，实际价格可能波动）：

| GPU型号 | 显存 | 参考价格（元/小时） | 适用场景 | 性价比 |
|--------|------|------------------|---------|-------|
| **RTX 3090** | 24GB | ~0.8-1.2 | 中小模型推理、微调 | ⭐⭐⭐⭐⭐ |
| **RTX 4090** | 24GB | ~1.3-1.8 | 中大模型推理、微调 | ⭐⭐⭐⭐ |
| **RTX 3080** | 10GB | ~0.5-0.8 | 小型模型实验 | ⭐⭐⭐⭐ |
| **V100** | 32GB | ~1.8-2.5 | 专业计算、训练 | ⭐⭐⭐ |
| **A100 40GB** | 40GB | ~3.0-4.0 | 大模型训练、推理 | ⭐⭐⭐⭐ |
| **A100 80GB** | 80GB | ~4.5-6.0 | 超大模型、长文本 | ⭐⭐⭐ |
| **A40** | 48GB | ~2.5-3.5 | 大显存需求场景 | ⭐⭐⭐⭐ |
| **RTX 4090D** | 24GB | ~1.2-1.6 | 4090替代方案 | ⭐⭐⭐⭐ |

> **注意**：实际价格因地域、时段和供需关系会有浮动。Spot实例价格可低至标价30%-50%。

**【截图位置 - 图2-1】**：实例创建页面，显示GPU类型选择列表和价格

---

### 2.2 如何选择地域

AutoDL 在全国多个地区设有数据中心：

| 地域 | 特点 | 推荐度 |
|------|------|-------|
| **北京** | GPU种类最全，A100/A40丰富 | ⭐⭐⭐⭐⭐ |
| **上海** | 网络质量好，适合华东用户 | ⭐⭐⭐⭐⭐ |
| **广东** | 价格相对较低，GPU充足 | ⭐⭐⭐⭐ |
| **成都** | 西部用户首选，性价比高 | ⭐⭐⭐⭐ |
| **香港** | 海外访问速度快 | ⭐⭐⭐ |

**选择建议：**
- 优先选择 **距离自己最近** 的地域，SSH延迟更低
- 如果需要特定GPU型号（如A100），选择该型号库存充足的地域
- 可以切换不同地域查看实时库存和价格

**【截图位置 - 图2-2】**：地域选择下拉菜单，显示可选地域列表

---

### 2.3 如何选择镜像

AutoDL 提供丰富的预配置镜像，推荐选择：

**推荐镜像路径（PyTorch）：**

1. 在创建实例页面，选择 **"镜像"** → **"算法镜像"**
2. 推荐选择以下镜像之一：

| 镜像名称 | PyTorch版本 | CUDA版本 | 适用场景 |
|---------|------------|---------|---------|
| PyTorch 2.3 + Python 3.10 | 2.3.x | CUDA 12.1 | 最新特性，推荐 |
| PyTorch 2.2 + Python 3.10 | 2.2.x | CUDA 12.1 | 稳定版本 |
| PyTorch 2.1 + Python 3.10 | 2.1.x | CUDA 12.1 | 兼容性最好 |

**操作步骤：**

1. 点击 **"算法镜像"** 标签
2. 在搜索框输入 "PyTorch"
3. 选择合适的版本（推荐 PyTorch 2.3+）
4. 确认包含 **CUDA** 和 **cuDNN**

> **提示**：选择镜像后，系统会自动安装好 PyTorch、CUDA 等基础环境，无需从零配置。

**【截图位置 - 图2-3】**：镜像选择界面，显示算法镜像列表和搜索框

---

### 2.4 存储配置

AutoDL 实例的存储结构：

| 存储类型 | 默认大小 | 用途 | 特点 |
|---------|---------|------|------|
| **系统盘** | 30-50GB | 操作系统、软件环境 | 随实例创建，速度较快 |
| **数据盘** | 可自定义50GB-2TB | 数据集、模型、代码 | 持久保存，实例释放后仍可保留 |

**配置建议：**

- **系统盘**：保持默认即可，安装基础软件
- **数据盘**：根据模型大小设置
  - 7B级别模型：建议 **50-100GB**
  - 13B-70B模型：建议 **200-500GB**
  - 多个大模型：建议 **500GB-1TB**

> **重要**：数据盘的内容在实例释放后仍可保留并挂载到新实例，是保存模型和数据的安全选择。

**【截图位置 - 图2-4】**：存储配置区域，显示系统盘和数据盘大小设置

---

### 2.5 推荐配置（按预算选择）

#### 经济型配置（推荐初学者）

| 项目 | 配置 |
|------|------|
| GPU | 1x RTX 3090 (24GB) |
| 地域 | 广东/成都（价格较低） |
| 镜像 | PyTorch 2.3 + Python 3.10 |
| 数据盘 | 100GB |
| 费用 | ~1元/小时，100元可跑约100小时 |
| 适用 | 7B模型推理、LoRA微调、QLoRA微调 |

#### 标准型配置（推荐常规研究）

| 项目 | 配置 |
|------|------|
| GPU | 1x RTX 4090 (24GB) |
| 地域 | 北京/上海 |
| 镜像 | PyTorch 2.3 + Python 3.10 |
| 数据盘 | 200GB |
| 费用 | ~1.5元/小时，200元可跑约130小时 |
| 适用 | 13B模型推理、8-bit量化、全参数微调7B模型 |

#### 高性能配置（大模型需求）

| 项目 | 配置 |
|------|------|
| GPU | 1x A100 40GB |
| 地域 | 北京 |
| 镜像 | PyTorch 2.3 + Python 3.10 |
| 数据盘 | 500GB |
| 费用 | ~3.5元/小时，200元可跑约57小时 |
| 适用 | 70B模型4-bit推理、大模型全参数微调 |

#### 多卡配置（分布式训练）

| 项目 | 配置 |
|------|------|
| GPU | 2x RTX 3090 或 2x A100 |
| 地域 | 北京/上海 |
| 镜像 | PyTorch 2.3 + Python 3.10 |
| 数据盘 | 500GB |
| 费用 | 约单卡价格的1.8-2倍 |
| 适用 | 模型并行、数据并行训练 |

**【截图位置 - 图2-5】**：实例创建完成页面，显示实例详情和连接信息

---

## 第三部分：环境配置

### 3.1 通过SSH连接实例

创建实例后，在控制台可以看到连接信息。AutoDL提供两种SSH连接方式：

**获取连接信息：**

1. 进入 AutoDL 控制台
2. 选择 **"容器实例"**
3. 找到已创建的实例，点击 **"SSH连接"** 或 **"登录指令"**
4. 记录以下信息：
   - 主机地址：`connect.example.autodl.com`（示例）
   - 端口号：`12345`（每台实例不同）
   - 用户名：`root`
   - 密码：实例创建时自动生成

**【截图位置 - 图3-1】**：控制台实例列表页面，显示SSH连接信息

---

#### Windows 用户（使用 PowerShell 或 Terminal）

**方法1：使用密码登录（推荐初学者）**

```powershell
# 打开 PowerShell 或 Windows Terminal
# 输入以下命令（替换为实际的主机和端口）
ssh -p 12345 root@connect.example.autodl.com

# 然后输入密码（密码不会显示在屏幕上，直接输入后回车）
```

**方法2：使用 SSH 密钥登录（推荐常用用户）**

```powershell
# 第一步：生成 SSH 密钥对（如果还没有）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 第二步：将公钥添加到 AutoDL
# 在 AutoDL 控制台 → "账号设置" → "公钥管理" 中添加公钥内容
# 公钥文件路径：C:\Users\你的用户名\.ssh\id_ed25519.pub

# 第三步：使用密钥登录（无需密码）
ssh -p 12345 root@connect.example.autodl.com
```

> **提示**：Windows 10/11 自带 OpenSSH 客户端，无需额外安装。如果提示找不到 ssh 命令，请安装 OpenSSH 客户端或 Git Bash。

---

#### Mac 用户

```bash
# 使用密码登录
ssh -p 12345 root@connect.example.autodl.com

# 或使用密钥登录（推荐）
# 先生成密钥（如果还没有）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 复制公钥到 AutoDL 控制台
# 公钥文件路径：~/.ssh/id_ed25519.pub
# 然后直接登录
ssh -p 12345 root@connect.example.autodl.com
```

---

#### Linux 用户

```bash
# 使用密码登录
ssh -p 12345 root@connect.example.autodl.com

# 或使用密钥登录
ssh -p 12345 -i ~/.ssh/id_ed25519 root@connect.example.autodl.com
```

---

#### 配置 SSH 快捷登录（推荐）

编辑本地 SSH 配置文件，简化登录：

**Windows/Mac/Linux 通用：**

```bash
# 编辑或创建 SSH 配置文件
# Linux/Mac: ~/.ssh/config
# Windows: C:\Users\用户名\.ssh\config
```

添加以下内容：

```
Host autodl
    HostName connect.example.autodl.com
    Port 12345
    User root
    # 如果使用密钥，取消下面这行注释
    # IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

配置完成后，可以直接使用：

```bash
ssh autodl
```

---

### 3.2 使用 JupyterLab

AutoDL 每个实例都内置 JupyterLab，可通过浏览器直接访问：

**访问方式：**

1. 在控制台找到实例，点击 **"JupyterLab"** 或 **"快捷工具"** → **"JupyterLab"**
2. 或者直接复制 "JupyterLab链接" 到浏览器打开
3. 链接格式示例：`http://connect.example.autodl.com:12346/`

**JupyterLab 功能：**

- **Terminal**：浏览器内打开终端，执行命令
- **Notebook**：创建和运行 Jupyter Notebook
- **文件管理**：上传/下载文件，管理目录
- **代码编辑器**：直接编辑代码文件

**【截图位置 - 图3-2】**：JupyterLab界面，显示左侧文件管理器和右侧终端

> **提示**：JupyterLab 的终端功能非常适合临时执行命令，但长时间运行的任务建议使用 SSH 连接后使用 `tmux` 或 `screen`。

---

### 3.3 上传代码和数据

AutoDL 提供多种文件传输方式：

#### 方法1：AutoDL 文件管理（最简单，适合小文件）

1. 打开 JupyterLab
2. 左侧文件管理器，点击 **"上传"** 按钮
3. 选择本地文件上传
4. 支持拖拽上传

**【截图位置 - 图3-3】**：JupyterLab文件管理器，显示上传按钮和文件列表

---

#### 方法2：FileZilla（图形界面，推荐Windows用户）

**下载安装：**

- 官网：https://www.filezilla.cn/download
- 下载 FileZilla Client（免费版）

**配置连接：**

1. 打开 FileZilla，点击 **"文件"** → **"站点管理器"**
2. 点击 **"新站点"**，输入名称如 "AutoDL"
3. 配置参数：
   - **协议**：SFTP - SSH File Transfer Protocol
   - **主机**：`connect.example.autodl.com`（替换为你的）
   - **端口**：`12345`（替换为你的SSH端口）
   - **登录类型**：正常
   - **用户**：`root`
   - **密码**：你的实例密码
4. 点击 **"连接"**

**使用方式：**

- 左侧：本地文件目录
- 右侧：远程服务器目录
- 拖拽文件即可上传/下载

**【截图位置 - 图3-4】**：FileZilla站点管理器配置界面

---

#### 方法3：scp 命令（命令行，适合Linux/Mac用户）

```bash
# 上传单个文件到远程
scp -P 12345 /本地/路径/file.py root@connect.example.autodl.com:/远程/路径/

# 上传整个文件夹（-r 递归）
scp -P 12345 -r /本地/路径/project_folder root@connect.example.autodl.com:/root/autodl-tmp/

# 从远程下载文件
scp -P 12345 root@connect.example.autodl.com:/远程/路径/result.txt /本地/路径/

# 从远程下载整个文件夹
scp -P 12345 -r root@connect.example.autodl.com:/root/autodl-tmp/results /本地/路径/
```

---

#### 方法4：rsync（大文件/大量文件，支持断点续传）

```bash
# 上传文件夹（显示进度，支持断点续传）
rsync -avz --progress -e "ssh -p 12345" /本地/路径/project/ root@connect.example.autodl.com:/root/autodl-tmp/project/

# 下载文件夹
rsync -avz --progress -e "ssh -p 12345" root@connect.example.autodl.com:/root/autodl-tmp/results/ /本地/路径/results/
```

> **推荐目录结构**：
> - 代码放在：`/root/autodl-tmp/project/`
> - 数据放在：`/root/autodl-tmp/data/`
> - 模型放在：`/root/autodl-tmp/models/`
> - 结果放在：`/root/autodl-tmp/results/`

---

### 3.4 配置 Conda 环境

AutoDL 预装了 Miniconda，可直接使用：

#### 查看已有环境

```bash
# 登录到实例后
conda env list

# 默认有一个 base 环境
```

#### 创建新环境（推荐）

```bash
# 创建名为 llm 的新环境，Python 3.10
conda create -n llm python=3.10 -y

# 激活环境
conda activate llm

# 验证 Python 版本
python --version
```

#### 环境管理常用命令

```bash
# 激活环境
conda activate llm

# 退出环境
conda deactivate

# 删除环境
conda remove -n llm --all

# 克隆环境
conda create -n llm_backup --clone llm

# 导出环境配置
conda env export > environment.yml

# 从配置文件创建环境
conda env create -f environment.yml
```

---

### 3.5 安装 PyTorch + Transformers + vLLM

#### 安装 PyTorch（GPU版本）

```bash
# 确保在 llm 环境中
conda activate llm

# 安装 PyTorch 2.3 + CUDA 12.1（推荐）
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 或者安装最新版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**验证 PyTorch 安装：**

```python
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'GPU数量: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

---

#### 安装 Transformers 及相关库

```bash
# 安装 transformers 和相关库
pip install transformers==4.41.0
pip install accelerate
pip install datasets
pip install peft                    # LoRA/QLoRA 微调
pip install bitsandbytes            # 4-bit/8-bit 量化
pip install sentencepiece           # 部分模型需要
pip install protobuf                # 部分模型需要
```

---

#### 安装 vLLM（高性能推理）

```bash
# vLLM 提供高效的 LLM 推理服务
# 注意：vLLM 对 PyTorch 版本有要求，请先确认 PyTorch 版本
pip install vllm==0.4.2

# 验证安装
python -c "import vllm; print(f'vLLM版本: {vllm.__version__}')"
```

> **注意**：vLLM 安装可能较慢，请耐心等待。如果遇到编译错误，可能需要更新 CUDA 版本或降级 vLLM 版本。

---

#### 其他常用库

```bash
# 数据处理和分析
pip install pandas numpy scipy scikit-learn matplotlib seaborn

# Jupyter（如果在SSH中需要运行notebook）
pip install jupyter ipykernel
python -m ipykernel install --user --name llm --display-name "Python (LLM)"

# 实验追踪
pip install wandb tensorboard

# 文本处理
pip install jieba                   # 中文分词

# API 服务
pip install fastapi uvicorn
```

---

### 3.6 配置 HuggingFace 镜像（国内下载加速）

由于网络原因，直接从 HuggingFace 下载模型可能很慢或失败。推荐以下加速方案：

#### 方案1：使用 HF-Mirror 镜像（推荐，最简单）

```bash
# 临时使用（当前终端有效）
export HF_ENDPOINT=https://hf-mirror.com

# 永久配置（写入 ~/.bashrc）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

#### 方案2：使用 ModelScope（国内平台，速度快）

```bash
# 安装 modelscope
pip install modelscope
```

**使用 ModelScope 下载模型：**

```python
from modelscope import snapshot_download

# 下载模型（自动映射到 ModelScope 的镜像）
model_path = snapshot_download(
    "Qwen/Qwen2-7B-Instruct",
    cache_dir="/root/autodl-tmp/models"
)
print(f"模型下载到: {model_path}")
```

#### 方案3：使用 AutoDL 学术加速（仅限 AutoDL 平台内）

AutoDL 提供了内置的网络加速，部分时段可用：

```bash
# 检查学术加速是否可用
autodl academic

# 开启学术加速（如果可用）
source /etc/network_turbo
```

> **提示**：`source /etc/network_turbo` 只在当前终端会话有效，需要在新终端中重新执行。

---

### 3.7 下载模型

#### 使用 HuggingFace + 镜像下载

```bash
# 先设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 使用 huggingface-cli 下载
huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2-7B-Instruct --local-dir /root/autodl-tmp/models/Qwen2-7B-Instruct
```

#### 使用 Python 脚本下载

```python
import os

# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定下载路径
cache_dir = "/root/autodl-tmp/models"

# 下载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    cache_dir=cache_dir,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    cache_dir=cache_dir,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)

print("模型下载完成！")
```

#### 使用 ModelScope 下载（推荐国内用户）

```python
from modelscope import snapshot_download

# 模型 ID 映射（HuggingFace -> ModelScope）
# 大多数模型在 ModelScope 上有同名镜像

model_name = "qwen/Qwen2-7B-Instruct"  # 注意 ModelScope 中可能大小写不同

cache_dir = "/root/autodl-tmp/models"
model_path = snapshot_download(
    model_name,
    cache_dir=cache_dir,
    revision="master"
)

print(f"模型下载到: {model_path}")
```

#### 常用模型下载参考

| 模型 | HuggingFace ID | ModelScope ID | 显存需求（4-bit） |
|------|---------------|---------------|----------------|
| Qwen2-7B | Qwen/Qwen2-7B-Instruct | qwen/Qwen2-7B-Instruct | ~6GB |
| Qwen2-72B | Qwen/Qwen2-72B-Instruct | qwen/Qwen2-72B-Instruct | ~40GB |
| ChatGLM3-6B | THUDM/chatglm3-6b | ZhipuAI/chatglm3-6b | ~5GB |
| Llama-3-8B | meta-llama/Meta-Llama-3-8B-Instruct | modelscope/Llama-3-8B-Instruct | ~7GB |
| Baichuan2-7B | baichuan-inc/Baichuan2-7B-Chat | baichuan-inc/Baichuan2-7B-Chat | ~6GB |
| Yi-6B | 01-ai/Yi-6B-Chat | 01-ai/Yi-6B-Chat | ~5GB |

---

## 第四部分：运行实验

### 4.1 上传实验代码

推荐使用以下方式组织项目：

```
/root/autodl-tmp/
├── project/                    # 代码目录
│   ├── run_inference.py       # 推理脚本
│   ├── run_finetune.py        # 微调脚本
│   ├── utils.py               # 工具函数
│   └── requirements.txt       # 依赖列表
├── data/                      # 数据目录
│   ├── train.json             # 训练数据
│   └── test.json              # 测试数据
├── models/                    # 模型目录（通过3.7下载）
│   └── Qwen2-7B-Instruct/    # 模型文件
└── results/                   # 输出结果
    ├── outputs/               # 模型输出
    └── logs/                  # 日志文件
```

**上传代码：**

```bash
# 本地终端执行（替换为你的连接信息）
scp -P 12345 -r ./project root@connect.example.autodl.com:/root/autodl-tmp/
```

---

### 4.2 安装依赖

```bash
# 登录到实例
ssh -p 12345 root@connect.example.autodl.com

# 激活环境
cd /root/autodl-tmp/project
conda activate llm

# 安装依赖
pip install -r requirements.txt
```

**requirements.txt 示例：**

```
torch>=2.3.0
transformers>=4.41.0
accelerate>=0.30.0
peft>=0.11.0
bitsandbytes>=0.43.0
datasets>=2.19.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
wandb>=0.17.0
```

---

### 4.3 运行实验

#### 使用 nohup 后台运行（简单场景）

```bash
# 后台运行脚本，输出到日志文件
nohup python run_inference.py > results/logs/inference.log 2>&1 &

# 查看进程
ps aux | grep python

# 实时查看日志
tail -f results/logs/inference.log
```

#### 使用 tmux 会话（推荐，可断开连接后继续运行）

```bash
# 创建新会话
tmux new -s experiment

# 在会话中运行实验
conda activate llm
cd /root/autodl-tmp/project
python run_inference.py

# 分离会话（保持后台运行）：按 Ctrl+B，然后按 D

# 重新连接到会话
tmux attach -t experiment

# 列出所有会话
tmux ls

# 结束会话
tmux kill-session -t experiment
```

**tmux 常用快捷键：**

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+B` 然后 `D` | 分离会话 |
| `Ctrl+B` 然后 `C` | 新建窗口 |
| `Ctrl+B` 然后 `N` | 下一个窗口 |
| `Ctrl+B` 然后 `P` | 上一个窗口 |
| `Ctrl+B` 然后 `"` | 水平分屏 |
| `Ctrl+B` 然后 `%` | 垂直分屏 |

---

#### 使用 screen（替代方案）

```bash
# 创建新会话
screen -S experiment

# 运行实验
python run_inference.py

# 分离会话：Ctrl+A 然后 D

# 重新连接
screen -r experiment

# 列出会话
screen -ls
```

---

### 4.4 监控GPU使用情况

#### nvidia-smi（查看当前GPU状态）

```bash
# 查看当前GPU状态
nvidia-smi

# 每秒刷新一次
nvidia-smi -l 1

# 查看详细信息
nvidia-smi -q

# 监控特定GPU
nvidia-smi -i 0  # 只显示第0号GPU
```

**nvidia-smi 输出说明：**

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        Off | 00000000:01:00.0 Off |                  N/A |
| 30%   55C    P2             220W / 350W |   8042MiB / 24576MiB |     85%      Default |
+-----------------------------------------+----------------------+----------------------+
```

**关键指标：**
- **GPU-Util**：GPU利用率（越高越好，接近100%说明充分利用）
- **Memory-Usage**：显存使用（注意不要超过上限）
- **Temp**：温度（低于85°C安全）
- **Pwr:Usage/Cap**：功耗使用/上限

---

#### gpustat（更美观的GPU监控）

```bash
# 安装
pip install gpustat

# 使用
gpustat

# 彩色显示，持续刷新
gpustat -cp --watch

# 显示进程信息
gpustat -cpu
```

---

#### nvtop（交互式GPU监控）

```bash
# 安装
apt update && apt install -y nvtop

# 运行（交互式界面）
nvtop
```

---

#### 在 Python 中监控

```python
import torch
import pynvml

def print_gpu_info():
    """打印GPU信息"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {props.name}")
            print(f"  显存: {allocated:.2f}GB / {props.total_memory / 1024**3:.2f}GB")
            print(f"  算力: {props.major}.{props.minor}")

# 调用
print_gpu_info()
```

---

### 4.5 处理常见问题

#### 显存不足（OOM - Out of Memory）

**症状：**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**解决方案：**

1. **使用量化加载（推荐）**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
```

2. **使用 8-bit 量化**

```python
model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)
```

3. **启用梯度检查点（训练时）**

```python
model.gradient_checkpointing_enable()
```

4. **减小批处理大小**

```python
# 减小 batch_size
batch_size = 1  # 从更大的值减小
```

5. **使用 CPU 卸载**

```python
model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    device_map="auto",  # 自动在GPU和CPU之间分配
    offload_folder="offload",
    trust_remote_code=True
)
```

---

#### 模型下载失败

**症状：** 下载中断、连接超时、校验失败

**解决方案：**

```bash
# 方案1：使用 --resume-download 断点续传
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download model_id --local-dir ./models

# 方案2：换用 ModelScope
# 修改代码使用 snapshot_download 从 ModelScope 下载

# 方案3：手动下载后上传
# 从 https://hf-mirror.com 网页手动下载模型文件
# 然后通过 scp/FileZilla 上传到实例

# 方案4：使用 aria2c 加速下载（多个小文件时）
apt install -y aria2
```

---

#### CUDA 版本不匹配

**症状：**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**解决方案：**

```bash
# 查看当前 CUDA 版本
nvcc --version
nvidia-smi  # 看右上角 CUDA Version

# 确保 PyTorch 的 CUDA 版本与系统一致
# 如果不一致，重新安装匹配版本的 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

---

### 4.6 保存结果到数据盘

**重要**：实例的系统盘（/root 下除 autodl-tmp 外）在实例重置时可能丢失，请务必将重要数据保存在数据盘。

```bash
# AutoDL 数据盘通常挂载在 /root/autodl-tmp/
# 这是持久化存储，实例释放后数据仍可保留

# 保存结果
cp -r /root/autodl-tmp/project/results /root/autodl-tmp/results_backup/

# 或者直接在代码中指定输出到数据盘
# 在 Python 脚本中：
# output_dir = "/root/autodl-tmp/results/experiment_001"
```

**自动备份脚本：**

```bash
#!/bin/bash
# save_results.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/root/autodl-tmp/backup/${TIMESTAMP}"

mkdir -p "${BACKUP_DIR}"

# 备份代码和结果
cp -r /root/autodl-tmp/project "${BACKUP_DIR}/"

# 打包压缩
cd /root/autodl-tmp/backup
tar czf "experiment_${TIMESTAMP}.tar.gz" "${TIMESTAMP}"

echo "备份完成: ${BACKUP_DIR}"
```

---

### 4.7 关机/释放实例

#### 临时关机（保存数据，不收费GPU）

```bash
# 在实例内执行关机（推荐先保存工作）
sudo shutdown now

# 或在 AutoDL 控制台点击 "关机" 按钮
```

> **注意**：关机后GPU不再计费，但数据盘费用可能继续（如有单独计费）。开机后可继续使用。

#### 释放实例（完全删除，数据盘可保留）

1. 确保所有重要数据已保存到数据盘
2. 在控制台选择实例，点击 **"释放"**
3. 选择是否保留数据盘
4. 确认释放

> **警告**：释放实例后不可恢复，务必确认数据已备份！

**【截图位置 - 图4-1】**：控制台实例操作菜单，显示关机和释放按钮

---

## 第五部分：成本优化

### 5.1 无卡模式开机（零费用配置环境）

AutoDL 提供 **"无卡模式"** 开机功能，可以在不使用GPU的情况下配置环境，大幅节省费用。

**使用场景：**
- 首次配置环境和安装依赖
- 上传和整理数据
- 调试代码（不需要GPU时）
- 下载模型文件

**操作步骤：**

1. 在控制台选择实例
2. 点击 **"更多操作"** → **"无卡模式开机"**
3. 实例启动后，可以通过 SSH/JupyterLab 连接
4. 此时只收取极低的CPU费用（约0.05-0.1元/小时）
5. 环境配置完成后，再切换回正常模式（使用GPU）

**【截图位置 - 图5-1】**：控制台显示"无卡模式开机"选项

---

### 5.2 随时关机释放

**最佳实践：**

| 场景 | 操作建议 |
|------|---------|
| 实验间隙休息 | 关机，保留实例 |
| 吃饭/开会（<1小时） | 可以保持开机，频繁开关反而麻烦 |
| 晚上睡觉 | 关机，第二天继续 |
| 周末不跑实验 | 关机 |
| 实验完全结束 | 释放实例，只保留数据盘 |

**快速操作：**

```bash
# 在实例内快速关机
sudo shutdown now

# 或使用 AutoDL 的快捷关机（在控制台操作更安全）
```

---

### 5.3 使用 Spot 实例（价格更低）

Spot 实例是利用平台空闲的GPU资源，价格通常比按量付费低 **30%-70%**，但可能被中断。

**适用场景：**
- 可以中断和恢复的实验
- 批量推理任务
- 模型训练（需要保存检查点）

**不适用场景：**
- 需要持续运行的服务
- 不能中断的关键实验

**使用方式：**

1. 创建实例时，在计费方式中选择 **"Spot"**
2. 设置最高出价（可选）
3. 创建实例

> **提示**：使用 Spot 实例时，建议定期保存检查点，以便被中断后可以从上次进度恢复。

**【截图位置 - 图5-2】**：实例创建页面的计费方式选择，显示"按量付费"和"Spot"选项

---

### 5.4 4-bit 量化节省显存

量化可以在几乎不损失效果的情况下大幅减少显存使用，从而可以使用更便宜的GPU。

**显存节省对比（以 Qwen2-7B 为例）：**

| 精度 | 显存占用 | 需要的GPU | 每小时成本 |
|------|---------|----------|-----------|
| FP16 | ~16GB | RTX 3090 | ~1元 |
| INT8 | ~8GB | RTX 3080 | ~0.6元 |
| INT4 | ~5GB | RTX 3080 | ~0.6元 |

**节省比例**：4-bit量化可节省约 **70%** 显存，对应约 **40%** 成本节省。

**量化代码示例：**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit 量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",  # 4-bit Normal Float，效果较好
    bnb_4bit_use_double_quant=True,  # 嵌套量化，进一步节省显存
)

model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/models/Qwen2-7B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
```

---

### 5.5 批量推理提高效率

**效率对比：**

| 推理方式 | 吞吐（tokens/sec） | 成本/千条样本 |
|---------|------------------|-------------|
| 单条循环 | 10-20 | 高 |
| 小批量（batch=8） | 80-150 | 中 |
| 大批量（batch=32） | 250-400 | 低 |
| vLLM 连续批处理 | 500-1000 | 最低 |

**使用 vLLM 批量推理：**

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="/root/autodl-tmp/models/Qwen2-7B-Instruct",
    tensor_parallel_size=1,  # GPU数量
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

# 准备提示
prompts = [
    "请分析以下经济数据：GDP增长率为5.2%，",
    "请解释通货膨胀对消费者的影响",
    "比较凯恩斯主义和新古典经济学的区别",
    # ... 更多提示
]

# 采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# 批量推理
outputs = llm.generate(prompts, sampling_params)

# 处理结果
for output in outputs:
    print(output.outputs[0].text)
```

> **提示**：vLLM 的连续批处理（continuous batching）可以自动优化GPU利用率，推荐用于生产级推理。

---

## 第六部分：代码示例

### 6.1 一键配置环境脚本

创建 `setup_env.sh` 文件，上传到实例后执行：

```bash
#!/bin/bash
# setup_env.sh - AutoDL 环境一键配置脚本
# 使用方法: chmod +x setup_env.sh && ./setup_env.sh

set -e  # 遇到错误立即退出

echo "========================================="
echo "AutoDL LLM 实验环境配置脚本"
echo "========================================="

# ========== 1. 设置环境变量 ==========
echo "[1/8] 设置环境变量..."

# 设置 HuggingFace 镜像
if ! grep -q "HF_ENDPOINT" ~/.bashrc; then
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
    echo "✓ HF_ENDPOINT 已添加到 ~/.bashrc"
fi
export HF_ENDPOINT=https://hf-mirror.com

# 设置常用目录
if ! grep -q "AUTODL_TMP" ~/.bashrc; then
    echo 'export AUTODL_TMP=/root/autodl-tmp' >> ~/.bashrc
    echo 'alias cdtmp="cd /root/autodl-tmp"' >> ~/.bashrc
    echo "✓ 常用变量已添加到 ~/.bashrc"
fi

source ~/.bashrc

# ========== 2. 创建工作目录 ==========
echo "[2/8] 创建工作目录..."
mkdir -p /root/autodl-tmp/{project,data,models,results,backup}
echo "✓ 工作目录创建完成"

# ========== 3. 创建 Conda 环境 ==========
echo "[3/8] 创建 Conda 环境 'llm'..."
if conda env list | grep -q "^llm "; then
    echo "✓ Conda 环境 'llm' 已存在，跳过创建"
else
    conda create -n llm python=3.10 -y
    echo "✓ Conda 环境 'llm' 创建完成"
fi

# 激活环境（在当前脚本中）
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm

# ========== 4. 安装 PyTorch ==========
echo "[4/8] 安装 PyTorch (CUDA 12.1)..."
pip install torch==2.3.1 torchvision==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121 -q
python -c "import torch; print(f'✓ PyTorch {torch.__version__} 安装完成, CUDA可用: {torch.cuda.is_available()}')"

# ========== 5. 安装 Transformers 生态 ==========
echo "[5/8] 安装 Transformers 及相关库..."
pip install -q \
    transformers==4.41.0 \
    accelerate>=0.30.0 \
    datasets>=2.19.0 \
    peft>=0.11.0 \
    bitsandbytes>=0.43.0 \
    sentencepiece \
    protobuf
python -c "import transformers; print(f'✓ Transformers {transformers.__version__} 安装完成')"

# ========== 6. 安装 vLLM ==========
echo "[6/8] 安装 vLLM..."
pip install vllm==0.4.2 -q 2>/dev/null || echo "⚠ vLLM 安装可能失败，请手动安装"
python -c "import vllm; print(f'✓ vLLM {vllm.__version__} 安装完成')" 2>/dev/null || echo "⚠ vLLM 未安装成功"

# ========== 7. 安装其他工具 ==========
echo "[7/8] 安装其他工具..."
pip install -q \
    pandas numpy scipy scikit-learn \
    matplotlib seaborn \
    jupyter ipykernel \
    wandb tensorboard \
    modelscope \
    gpustat

# 注册 Jupyter 内核
python -m ipykernel install --user --name llm --display-name "Python (LLM)" 2>/dev/null || true

# 安装系统工具
apt update -qq && apt install -y -qq nvtop htop tree aria2 2>/dev/null || true

echo "✓ 其他工具安装完成"

# ========== 8. 验证安装 ==========
echo "[8/8] 验证安装..."
echo ""
echo "========================================="
echo "安装验证结果："
echo "========================================="

python << 'EOF'
import sys
print(f"Python版本: {sys.version}")

try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
except Exception as e:
    print(f"PyTorch检查失败: {e}")

try:
    import transformers
    print(f"Transformers版本: {transformers.__version__}")
except:
    print("Transformers: 未安装")

try:
    import vllm
    print(f"vLLM版本: {vllm.__version__}")
except:
    print("vLLM: 未安装")

try:
    import peft
    print(f"PEFT版本: {peft.__version__}")
except:
    print("PEFT: 未安装")

print("=========================================")
EOF

echo ""
echo "========================================="
echo "✓ 环境配置完成！"
echo "========================================="
echo ""
echo "使用说明："
echo "  激活环境: conda activate llm"
echo "  进入工作目录: cd /root/autodl-tmp"
echo "  查看GPU: nvidia-smi 或 gpustat"
echo ""
echo "目录结构："
echo "  /root/autodl-tmp/project - 代码"
echo "  /root/autodl-tmp/data    - 数据"
echo "  /root/autodl-tmp/models  - 模型"
echo "  /root/autodl-tmp/results - 结果"
echo "========================================="
```

**使用方法：**

```bash
# 1. 上传脚本到实例
scp -P 12345 setup_env.sh root@connect.example.autodl.com:/root/

# 2. SSH 登录实例后执行
ssh -p 12345 root@connect.example.autodl.com
chmod +x /root/setup_env.sh
./setup_env.sh

# 整个配置过程约 5-10 分钟
```

---

### 6.2 模型推理示例

```python
#!/usr/bin/env python3
# run_inference.py - 使用 Qwen2 进行推理

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ========== 配置 ==========
MODEL_PATH = "/root/autodl-tmp/models/Qwen2-7B-Instruct"
DATA_PATH = "/root/autodl-tmp/data/test_prompts.json"
OUTPUT_PATH = "/root/autodl-tmp/results/inference_results.json"
USE_4BIT = True  # 使用4-bit量化节省显存
MAX_NEW_TOKENS = 512

# 设置HuggingFace镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ========== 加载模型 ==========
def load_model():
    print("正在加载模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    
    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    print("模型加载完成！")
    return model, tokenizer

# ========== 推理函数 ==========
def generate_response(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "你是一个专业的经济学研究助手。"},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]
    
    return response

# ========== 主函数 ==========
def main():
    # 加载模型
    model, tokenizer = load_model()
    
    # 读取输入数据
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = data if isinstance(data, list) else data.get("prompts", [])
    else:
        # 使用默认提示
        prompts = [
            "请分析2024年中国GDP增长的主要驱动因素。",
            "解释货币政策和财政政策在经济调控中的区别和联系。",
            "比较绝对优势理论和比较优势理论的异同。"
        ]
    
    # 执行推理
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n处理 [{i+1}/{len(prompts)}]: {prompt[:50]}...")
        try:
            response = generate_response(model, tokenizer, prompt)
            results.append({
                "prompt": prompt,
                "response": response,
                "status": "success"
            })
            print(f"回答: {response[:200]}...")
        except Exception as e:
            results.append({
                "prompt": prompt,
                "response": str(e),
                "status": "error"
            })
            print(f"错误: {e}")
    
    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 推理完成！结果已保存到: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
```

---

### 6.3 批量推理示例（使用 vLLM）

```python
#!/usr/bin/env python3
# run_batch_inference.py - 使用 vLLM 批量推理

import os
import json
from vllm import LLM, SamplingParams

# ========== 配置 ==========
MODEL_PATH = "/root/autodl-tmp/models/Qwen2-7B-Instruct"
DATA_PATH = "/root/autodl-tmp/data/batch_prompts.json"
OUTPUT_PATH = "/root/autodl-tmp/results/batch_results.json"
BATCH_SIZE = 32

# ========== 主函数 ==========
def main():
    print("正在加载模型...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )
    
    # 加载提示
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data if isinstance(data, list) else data.get("prompts", [])
    
    print(f"共 {len(prompts)} 条提示，开始批量推理...")
    
    # 批量推理
    outputs = llm.generate(prompts, sampling_params)
    
    # 整理结果
    results = []
    for i, output in enumerate(outputs):
        results.append({
            "prompt": prompts[i],
            "response": output.outputs[0].text,
        })
    
    # 保存
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 批量推理完成！结果已保存到: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
```

---

### 6.4 QLoRA 微调示例

```python
#!/usr/bin/env python3
# run_qlora_finetune.py - QLoRA 微调示例

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# ========== 配置 ==========
MODEL_PATH = "/root/autodl-tmp/models/Qwen2-7B-Instruct"
DATA_PATH = "/root/autodl-tmp/data/train.json"
OUTPUT_DIR = "/root/autodl-tmp/results/qlora_output"

# LoRA 配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 训练配置
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ========== 主函数 ==========
def main():
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("加载模型（4-bit量化）...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 配置
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载数据集
    print("加载数据集...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="none",
    )
    
    # 训练
    print("开始训练...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    
    trainer.train()
    
    # 保存模型
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    print(f"✓ 训练完成！模型已保存到: {OUTPUT_DIR}/final_model")

if __name__ == "__main__":
    main()
```

---

## 第七部分：故障排除

### 7.1 SSH 连接不上

| 问题现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `Connection refused` | 实例未开机 | 在控制台确认实例状态为"运行中" |
| `Connection timed out` | 网络问题或端口错误 | 检查端口是否正确，尝试切换网络 |
| `Permission denied` | 密码错误或密钥不匹配 | 重新复制密码，检查密钥配置 |
| `Host key verification failed` | 服务器指纹变化 | 删除 `~/.ssh/known_hosts` 中对应行 |
| 连接后立即断开 | 实例资源不足 | 检查实例状态，可能需要重启 |

**排查步骤：**

```bash
# 1. 检查实例是否运行
# 登录 AutoDL 控制台，确认实例状态为"运行中"

# 2. 检查网络连通性（本地执行）
ping connect.example.autodl.com

# 3. 检查端口是否开放（本地执行）
telnet connect.example.autodl.com 12345
# 或
nc -vz connect.example.autodl.com 12345

# 4. 使用详细模式查看错误（本地执行）
ssh -v -p 12345 root@connect.example.autodl.com

# 5. 如果密钥有问题，尝试密码登录
ssh -o PreferredAuthentications=password -p 12345 root@connect.example.autodl.com
```

---

### 7.2 显存不足（OOM）

**完整解决方案清单：**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ========== 方案1: 4-bit 量化（推荐） ==========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    quantization_config=bnb_config,
    device_map="auto",
)

# ========== 方案2: 8-bit 量化 ==========
model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    load_in_8bit=True,
    device_map="auto",
)

# ========== 方案3: 使用更小的数据类型 ==========
model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    torch_dtype=torch.float16,  # 不使用 float32
    device_map="auto",
)

# ========== 方案4: 梯度检查点（训练时） ==========
model.gradient_checkpointing_enable()

# ========== 方案5: 减小序列长度 ==========
MAX_SEQ_LENGTH = 256  # 从 512/1024 减小

# ========== 方案6: 减小批大小 ==========
BATCH_SIZE = 1  # 减小到最小

# ========== 方案7: 使用 DeepSpeed ZeRO-Offload ==========
# 安装: pip install deepspeed
# 配置文件 zero_config.json:
# {
#   "bf16": {"enabled": true},
#   "zero_optimization": {
#     "stage": 2,
#     "offload_optimizer": {"device": "cpu"}
#   }
# }
```

**显存占用估算：**

| 模型大小 | FP16 | 8-bit | 4-bit |
|---------|------|-------|-------|
| 7B | ~14GB | ~7GB | ~4GB |
| 13B | ~26GB | ~13GB | ~7GB |
| 70B | ~140GB | ~70GB | ~40GB |

> **提示**：推理时额外需要约 2-4GB 用于 KV Cache，训练时额外需要约 2-3 倍模型大小的显存用于优化器状态。

---

### 7.3 模型下载慢/失败

**排查清单：**

```bash
# 1. 检查是否设置了镜像
export HF_ENDPOINT=https://hf-mirror.com
echo $HF_ENDPOINT

# 2. 测试镜像连通性
curl -I https://hf-mirror.com

# 3. 使用 aria2c 多线程下载（大文件）
apt install -y aria2
aria2c -x 4 -s 4 "https://hf-mirror.com/.../model.bin"

# 4. 手动下载后上传
# 从 https://hf-mirror.com 找到模型页面
# 手动下载每个文件，然后通过 scp 上传

# 5. 使用 ModelScope 替代
# 修改代码使用 modelscope 的 snapshot_download

# 6. 使用学术加速（AutoDL内）
source /etc/network_turbo

# 7. 检查磁盘空间
df -h
# 确保数据盘有足够空间（至少模型大小的2倍）

# 8. 断点续传
huggingface-cli download --resume-download model_id
```

---

### 7.4 依赖冲突

**常见问题：**

```bash
# 问题: transformers 和 torch 版本不兼容
# 解决: 安装匹配的版本
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.41.0

# 问题: CUDA 版本不匹配
# 检查:
nvcc --version  # 系统 CUDA 版本
python -c "import torch; print(torch.version.cuda)"  # PyTorch 编译的 CUDA 版本

# 如果不匹配，重新安装 PyTorch:
pip uninstall torch torchvision torchaudio -y
pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# 问题: numpy 版本太新导致不兼容
# 解决:
pip install numpy==1.24.3  # 降级到稳定版本

# 问题: protobuf 版本冲突
# 解决:
pip install protobuf==3.20.3  # 使用稳定版本

# 通用解决方案：干净环境
conda create -n llm_clean python=3.10 -y
conda activate llm_clean
pip install torch transformers accelerate -q  # 重新安装核心依赖
```

---

### 7.5 权限问题

```bash
# 问题: Permission denied 执行脚本
# 解决:
chmod +x script.sh

# 问题: pip install 权限不足
# 解决: 使用 --user 参数或在conda环境中安装
pip install --user package_name
# 或
conda activate llm
pip install package_name

# 问题: 无法写入目录
# 解决: 检查目录权限
ls -la /root/autodl-tmp/
chmod 755 /root/autodl-tmp/project

# 问题: conda 命令找不到
# 解决:
source ~/miniconda3/etc/profile.d/conda.sh
# 或
export PATH="~/miniconda3/bin:$PATH"
```

---

### 7.6 其他常见问题

#### JupyterLab 无法访问

1. 确认实例已开机
2. 检查 JupyterLab 链接是否正确（注意端口号）
3. 尝试清除浏览器缓存或换浏览器
4. 检查实例内的 Jupyter 服务是否运行

#### GPU 不被识别

```bash
# 检查驱动
nvidia-smi

# 如果报错，可能需要重新安装驱动（联系客服）
# 或尝试重启实例

# 检查 PyTorch 是否能识别
torch.cuda.is_available()
# 如果返回 False，可能是 CUDA 版本不匹配
```

#### 实例意外关机

1. 检查余额是否充足
2. 检查是否使用了 Spot 实例（可能被回收）
3. 检查实例是否设置了定时关机
4. 联系 AutoDL 客服

---

## 附录

### A. 快捷键和常用命令速查

| 命令/操作 | 说明 |
|----------|------|
| `ssh -p PORT root@HOST` | SSH连接 |
| `scp -P PORT file root@HOST:/path/` | 上传文件 |
| `tmux new -s NAME` | 创建tmux会话 |
| `tmux attach -t NAME` | 连接tmux会话 |
| `nvidia-smi` | 查看GPU状态 |
| `gpustat -cp` | 美观的GPU监控 |
| `df -h` | 查看磁盘空间 |
| `du -sh /path` | 查看目录大小 |
| `conda activate llm` | 激活conda环境 |
| `pip list` | 查看已安装包 |
| `top` / `htop` | 查看进程 |
| `ps aux \| grep python` | 查看Python进程 |
| `kill -9 PID` | 终止进程 |

### B. 推荐资源

- **AutoDL 官方文档**：https://www.autodl.com/docs
- **PyTorch 官方文档**：https://pytorch.org/docs
- **Transformers 文档**：https://huggingface.co/docs/transformers
- **HuggingFace 镜像**：https://hf-mirror.com
- **ModelScope 社区**：https://www.modelscope.cn

### C. 费用估算表

| GPU | 单价（元/小时） | 100元可用时长 | 适合模型 |
|-----|---------------|-------------|---------|
| RTX 3080 | ~0.6 | ~166小时 | 7B 4-bit |
| RTX 3090 | ~1.0 | ~100小时 | 7B-13B 4-bit |
| RTX 4090 | ~1.5 | ~66小时 | 13B-70B 4-bit |
| A100 40GB | ~3.5 | ~28小时 | 70B 4-bit |
| A100 80GB | ~5.0 | ~20小时 | 大模型全精度 |

> **注意**：以上价格为按量付费参考价，Spot实例价格可能更低。

---

> **文档版本**：2025年6月  
> **作者**：AutoDL GPU云平台使用指南  
> **适用平台**：AutoDL (https://www.autodl.com)

---
*本文档为学术研究用途编写，价格和政策可能随时间变化，请以 AutoDL 官网最新信息为准。*
