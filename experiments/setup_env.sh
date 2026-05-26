#!/bin/bash
# setup_env.sh - AutoDL环境一键配置脚本
# 使用方法: bash setup_env.sh
#
# 功能说明:
#   1. 自动创建conda环境 (Python 3.10)
#   2. 根据AutoDL实例的CUDA版本自动安装匹配的PyTorch
#   3. 安装transformers、datasets、accelerate等核心依赖
#   4. 安装vLLM加速推理库
#   5. 配置HuggingFace国内镜像加速模型下载
#   6. 创建标准实验目录结构
#
# 适用平台: AutoDL (https://www.autodl.com)
# 测试环境: CUDA 11.8 / CUDA 12.1, Ubuntu 20.04/22.04

set -e  # 遇到错误立即退出

# ============ 彩色输出 ============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

# ============ 1. 创建conda环境 ============
info "步骤 1/6: 创建conda环境 'info_depreciation' (Python 3.10)"
conda create -n info_depreciation python=3.10 -y
source activate info_depreciation
ok "conda环境创建完成"

# ============ 2. 安装PyTorch（根据AutoDL的CUDA版本自动选择） ============
info "步骤 2/6: 检测CUDA版本并安装对应PyTorch"

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    info "检测到CUDA版本: $CUDA_VERSION"
else
    warn "未检测到nvcc，尝试使用nvidia-smi获取CUDA版本"
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p')
    info "nvidia-smi检测到CUDA版本: $CUDA_VERSION"
fi

if [[ "$CUDA_VERSION" == "12.1" ]]; then
    info "安装 PyTorch 2.1.0 (CUDA 12.1)"
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "12.4" ]] || [[ "$CUDA_VERSION" == "12.3" ]]; then
    warn "CUDA $CUDA_VERSION 无精确匹配，使用cu121版本"
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    info "安装 PyTorch 2.1.0 (CUDA 11.8)"
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
else
    warn "未检测到标准CUDA版本($CUDA_VERSION)，默认使用CUDA 11.8版本"
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
fi
ok "PyTorch安装完成"

# ============ 3. 安装核心依赖 ============
info "步骤 3/6: 安装transformers及相关库"
pip install transformers==4.35.0
pip install datasets==2.14.0
pip install accelerate==0.24.0
pip install bitsandbytes==0.41.0
pip install scipy numpy tqdm pandas matplotlib
ok "核心依赖安装完成"

# ============ 4. 安装vLLM（可选加速） ============
info "步骤 4/6: 安装vLLM加速推理库"
pip install vllm==0.2.1 || warn "vLLM安装失败，将使用HuggingFace原生推理"
ok "vLLM处理完成"

# ============ 5. 配置HuggingFace国内镜像 ============
info "步骤 5/6: 配置HuggingFace镜像（国内加速）"

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

# ============ 6. 创建实验目录 ============
info "步骤 6/6: 创建实验目录结构"
mkdir -p /root/autodl-tmp/experiments/{results,models,data,logs}
cd /root/autodl-tmp/experiments

# 创建README
ok "实验目录结构创建完成"

# ============ 完成提示 ============
echo ""
echo "========================================"
echo -e "${GREEN}     环境配置全部完成！${NC}"
echo "========================================"
echo ""
echo -e "  conda环境: ${YELLOW}info_depreciation${NC}"
echo -e "  Python版本: ${YELLOW}3.10${NC}"
echo -e "  CUDA版本: ${YELLOW}$CUDA_VERSION${NC}"
echo -e "  PyTorch: ${YELLOW}2.1.0${NC}"
echo -e "  实验目录: ${YELLOW}/root/autodl-tmp/experiments/${NC}"
echo -e "  模型缓存: ${YELLOW}/root/autodl-tmp/huggingface_cache/${NC}"
echo ""
echo -e "  激活环境: ${GREEN}conda activate info_depreciation${NC}"
echo -e "  查看帮助: ${GREEN}python run_experiments.py --help${NC}"
echo ""
echo "========================================"
