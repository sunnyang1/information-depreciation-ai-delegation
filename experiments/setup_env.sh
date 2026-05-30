#!/bin/bash
# setup_env.sh - AutoDL环境一键配置脚本 (v3.1)
# 更新记录:
#   v3.1 - 适配 RTX 5090 (Blackwell) + PyTorch 2.7.0 + CUDA 12.8 + Python 3.12
#   v3.0 - 适配重构后的实验架构：统一注册入口(registry.py) + 独立实验模块(exps/)
#   v2.1 - 修复 datasets==2.14.0 与 pyarrow>=15 的兼容性问题
# 使用方法: bash setup_env.sh
#
# 功能说明:
#   1. 检测预装环境 (Python 3.12 / PyTorch 2.7.0 / CUDA 12.8)
#   2. 如未预装则创建conda环境并安装PyTorch
#   3. 安装transformers、datasets、accelerate等核心依赖
#   4. 安装vLLM加速推理库（Blackwell架构自动适配）
#   5. 安装statsmodels（回归分析所需）
#   6. 配置HuggingFace国内镜像加速模型下载
#   7. 创建标准实验目录结构
#   8. 环境验证
#
# 适用平台: AutoDL (https://www.autodl.com)
# 测试环境: CUDA 12.8, PyTorch 2.7.0, Python 3.12, RTX 5090 (Blackwell)

set -e  # 遇到错误立即退出

# ============ 彩色输出 ============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }
highlight() { echo -e "${CYAN}$1${NC}"; }

# ============ 辅助函数：检测预装PyTorch ============
check_pytorch_installed() {
    python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none"
}

# ============ 辅助函数：检测GPU架构 ============
detect_gpu_arch() {
    python -c "
import subprocess, sys
try:
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    stdout = result.stdout.lower()
    if 'rtx 50' in stdout or 'rtx 5090' in stdout or 'rtx 5080' in stdout:
        print('blackwell')
    elif 'rtx 40' in stdout or 'rtx 4090' in stdout or 'rtx 3090' in stdout:
        print('ada_lovelace')
    elif 'a100' in stdout or 'a800' in stdout:
        print('ampere')
    elif 'h100' in stdout or 'h800' in stdout or 'h200' in stdout:
        print('hopper')
    else:
        print('unknown')
except Exception:
    print('unknown')
" 2>/dev/null || echo "unknown"
}

# ============ 0. 克隆项目（如尚未克隆） ============
PROJECT_DIR="/root/autodl-tmp/MS_Attention4Org"
if [ ! -d "$PROJECT_DIR" ]; then
    info "步骤 0/10: 从GitHub克隆项目仓库"
    mkdir -p /root/autodl-tmp
    cd /root/autodl-tmp
    git clone https://github.com/sunnyang1/information-depreciation-ai-delegation.git "$PROJECT_DIR" || warn "克隆失败，请手动上传代码"
fi

# ============ 1. 检测预装环境 ============
info "步骤 1/10: 检测预装环境"

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTORCH_VERSION=$(check_pytorch_installed)
GPU_ARCH=$(detect_gpu_arch)

info "Python版本: $PYTHON_VERSION"
info "PyTorch版本: $PYTORCH_VERSION"
info "GPU架构: $GPU_ARCH"

if [ "$GPU_ARCH" = "blackwell" ]; then
    highlight "检测到 Blackwell 架构 GPU (RTX 50系列)"
fi

# ============ 2. 创建/激活conda环境 ============
info "步骤 2/10: 配置conda环境"

if [ "$PYTHON_VERSION" = "3.12" ] && [ "$PYTORCH_VERSION" != "none" ]; then
    ok "检测到预装 Python 3.12 + PyTorch，跳过conda环境创建"
    ENV_TYPE="prebuilt"
else
    info "未检测到预装环境，创建conda环境 'info_depreciation' (Python 3.12)"
    conda create -n info_depreciation python=3.12 -y
    source activate info_depreciation
    ok "conda环境创建完成"
    ENV_TYPE="conda"
fi

# ============ 3. 安装/验证PyTorch ============
info "步骤 3/10: 安装/验证PyTorch"

if [ "$PYTORCH_VERSION" != "none" ]; then
    ok "PyTorch 已预装 (v$PYTORCH_VERSION)，跳过安装"
    
    # 验证CUDA可用性
    python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA可用: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA不可用 (CPU模式)')
"
else
    info "未检测到PyTorch，开始安装"
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        info "检测到CUDA版本: $CUDA_VERSION"
    else
        warn "未检测到nvcc，尝试使用nvidia-smi获取CUDA版本"
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p')
        info "nvidia-smi检测到CUDA版本: $CUDA_VERSION"
    fi
    
    if [[ "$CUDA_VERSION" == "12.8" ]] || [[ "$CUDA_VERSION" == "12.6" ]] || [[ "$CUDA_VERSION" == "12.5" ]] || [[ "$CUDA_VERSION" == "12.4" ]]; then
        info "安装 PyTorch 2.7.0 (CUDA 12.8 compatible)"
        pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    elif [[ "$CUDA_VERSION" == "12.1" ]]; then
        info "安装 PyTorch 2.7.0 (CUDA 12.1)"
        pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == "11.8" ]]; then
        info "安装 PyTorch 2.7.0 (CUDA 11.8)"
        pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
    else
        warn "未检测到标准CUDA版本($CUDA_VERSION)，默认使用CUDA 12.1版本"
        pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu121
    fi
    ok "PyTorch安装完成"
fi

# ============ 4. 安装核心依赖 ============
info "步骤 4/10: 安装transformers及相关库"

# 统一使用 transformers>=4.40.0，支持 Qwen2.5、Llama-3.1 等新架构
pip install transformers>=4.40.0
pip install tokenizers

# pyarrow 兼容性处理
pip install "pyarrow>=12.0.0,<14.0.0"
pip install datasets==2.14.0
pip install accelerate>=0.24.0
pip install scipy numpy tqdm pandas matplotlib statsmodels
ok "核心依赖安装完成"

# ============ 5. 安装量化与加速库 ============
info "步骤 5/10: 安装量化与加速库"

# bitsandbytes: Blackwell需要新版本
if [ "$GPU_ARCH" = "blackwell" ]; then
    info "Blackwell架构：安装新版 bitsandbytes"
    pip install bitsandbytes>=0.43.0 || warn "新版 bitsandbytes 安装失败"
else
    pip install bitsandbytes==0.41.0 || warn "bitsandbytes安装失败"
fi
ok "量化库处理完成"

# ============ 6. 安装vLLM ============
info "步骤 6/10: 安装vLLM加速推理库"

if [ "$GPU_ARCH" = "blackwell" ]; then
    warn "Blackwell架构 (RTX 5090) 下 vLLM 0.2.1 不兼容"
    info "尝试安装新版 vLLM (>=0.11.0) ..."
    pip install vllm>=0.11.0 || warn "新版vLLM安装失败，将使用HuggingFace原生推理"
else
    pip install vllm==0.2.1 || warn "vLLM安装失败，将使用HuggingFace原生推理"
fi
ok "vLLM处理完成"

# ============ 7. 配置HuggingFace国内镜像 ============
info "步骤 7/10: 配置HuggingFace镜像（国内加速）"

# 创建HuggingFace配置文件
mkdir -p ~/.config/huggingface
cat > ~/.config/huggingface/config.yaml << 'EOF'
hf_endpoint: https://hf-mirror.com
EOF

# 设置环境变量（当前会话 + 永久生效）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface_cache

# 写入.bashrc实现永久生效
if ! grep -q "HF_ENDPOINT" ~/.bashrc; then
    echo '' >> ~/.bashrc
    echo '# HuggingFace镜像配置 (由setup_env.sh添加)' >> ~/.bashrc
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
    echo 'export HF_HOME=/root/autodl-tmp/huggingface_cache' >> ~/.bashrc
fi

# 创建模型缓存目录
mkdir -p /root/autodl-tmp/huggingface_cache

ok "HF镜像配置完成 (使用 hf-mirror.com)"

# ============ 8. 创建实验目录 ============
info "步骤 8/10: 创建实验目录结构"
mkdir -p "$PROJECT_DIR/experiments"/{results,figures,models,data,logs,checkpoints}
ok "实验目录结构创建完成"

# ============ 9. 环境验证 ============
info "步骤 9/10: 验证关键包安装"

python << 'PYEOF'
import sys
errors = []

def check(pkg):
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False

packages = [
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("datasets", "Datasets"),
    ("accelerate", "Accelerate"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("pandas", "Pandas"),
    ("matplotlib", "Matplotlib"),
    ("statsmodels", "Statsmodels"),
]

optional = [
    ("vllm", "vLLM"),
    ("bitsandbytes", "bitsandbytes"),
]

for mod, name in packages:
    if check(mod):
        print(f"  [OK] {name}")
    else:
        print(f"  [FAIL] {name}")
        errors.append(name)

for mod, name in optional:
    if check(mod):
        print(f"  [OK] {name} (optional)")
    else:
        print(f"  [WARN] {name} (optional, not installed)")

if errors:
    print(f"\n[ERROR] 以下关键包未安装成功: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\n[OK] 所有关键包验证通过")
PYEOF

ok "环境验证完成"

# ============ 10. 创建快捷脚本 ============
info "步骤 10/10: 创建实验运行快捷脚本"

EXP_DIR="$PROJECT_DIR/experiments"

cat > "$EXP_DIR/run_simulation.sh" << 'EOF'
#!/bin/bash
# 一键运行全部模拟实验（无需GPU）
set -e
cd "$(dirname "$0")"
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    : # 已在conda环境中
else
    conda activate info_depreciation 2>/dev/null || source activate info_depreciation
fi

echo "========================================"
echo "  运行模拟实验 (Simulation Mode)"
echo "========================================"

echo ""
echo "[1/3] 列出所有已注册实验..."
python run.py --list

echo ""
echo "[2/3] 运行全部实验..."
python run.py --experiment all --output results

echo ""
echo "[3/3] 运行完成。也可按类别运行:"
echo "  python run.py --category baseline          # 基础实验"
echo "  python run.py --category advanced          # 进阶实验"
echo "  python run.py --category reviewer          # 审稿人回复实验"
echo "  python run.py --category v6_architecture   # V6架构实验"
echo "  python run.py --category v6_frontloading   # V6前置加载实验"
echo "  python run.py --category v6_microfoundation # V6微观基础实验"
echo "  python run.py --category supplementary     # 补充图表"
echo ""
echo "========================================"
echo "  模拟实验全部完成"
echo "  结果: results/"
echo "  图表: figures/"
echo "========================================"
EOF
chmod +x "$EXP_DIR/run_simulation.sh"

cat > "$EXP_DIR/run_real.sh" << 'EOF'
#!/bin/bash
# 一键运行真实LLM实验（需要GPU）
set -e
cd "$(dirname "$0")"
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    : # 已在conda环境中
else
    conda activate info_depreciation 2>/dev/null || source activate info_depreciation
fi

echo "========================================"
echo "  运行真实LLM实验 (Real Inference)"
echo "========================================"

# 默认使用 Llama-2-7B，可修改 --model_size 参数
python run_real_experiments.py --experiment all --model_size 7b "$@"

echo ""
echo "========================================"
echo "  真实LLM实验全部完成"
echo "  结果: results/"
echo "========================================"
EOF
chmod +x "$EXP_DIR/run_real.sh"

cat > "$EXP_DIR/run_single.sh" << 'EOF'
#!/bin/bash
# 运行单个实验（用法: bash run_single.sh exp01）
set -e
cd "$(dirname "$0")"
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    : # 已在conda环境中
else
    conda activate info_depreciation 2>/dev/null || source activate info_depreciation
fi

EXP_ID="${1:-exp01}"
echo "========================================"
echo "  运行单个实验: $EXP_ID"
echo "========================================"

python run.py --experiment "$EXP_ID" --output results

echo "========================================"
EOF
chmod +x "$EXP_DIR/run_single.sh"

ok "快捷脚本创建完成"

# ============ 完成提示 ============
echo ""
echo "========================================"
echo -e "${GREEN}     环境配置全部完成！${NC}"
echo "========================================"
echo ""
echo -e "  conda环境: ${YELLOW}info_depreciation${NC}"
echo -e "  Python版本: ${YELLOW}$(python --version | awk '{print $2}')${NC}"
echo -e "  PyTorch版本: ${YELLOW}$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')${NC}"
echo -e "  CUDA版本: ${YELLOW}$(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'N/A')${NC}"
echo -e "  GPU架构: ${YELLOW}$GPU_ARCH${NC}"
echo -e "  实验目录: ${YELLOW}$PROJECT_DIR/experiments/${NC}"
echo -e "  模型缓存: ${YELLOW}/root/autodl-tmp/huggingface_cache/${NC}"
echo ""
echo -e "  激活环境: ${GREEN}conda activate info_depreciation${NC}"
echo ""
echo -e "  ${BLUE}模拟实验 (无需GPU):${NC}"
echo -e "    ${GREEN}bash $PROJECT_DIR/experiments/run_simulation.sh${NC}"
echo -e "    或手动: cd $PROJECT_DIR/experiments && python run.py --experiment all"
echo -e "    按类别: cd $PROJECT_DIR/experiments && python run.py --category baseline"
echo -e "    单实验: cd $PROJECT_DIR/experiments && python run.py --experiment exp01"
echo -e "    列全部: cd $PROJECT_DIR/experiments && python run.py --list"
echo ""
echo -e "  ${BLUE}真实LLM实验 (需要GPU):${NC}"
echo -e "    ${GREEN}bash $PROJECT_DIR/experiments/run_real.sh${NC}"
echo -e "    或手动: cd $PROJECT_DIR/experiments && python run_real_experiments.py --experiment all --model_size 7b"
echo ""
echo -e "  ${BLUE}回归分析:${NC}"
echo -e "    ${GREEN}cd $PROJECT_DIR && python simulation_regression.py${NC}"
echo ""
echo "========================================"
