#!/bin/bash
# setup_env_v3_vllm09.sh - vLLM 0.9.2 + PyTorch 2.7.0 + CUDA 12.8
# 推荐GPU: RTX 5090 (Blackwell) / A100 / H100
# 注意: vLLM 0.9.2 比 0.9.0 更稳定(修复了numerical error和FusedMoE bug)
# RTX 5090 需要 CUDA 12.8 + PyTorch 2.7 + vLLM 0.9+ 才能支持 Blackwell 架构

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

info "步骤 1/6: 创建conda环境"
conda create -n info_depreciation python=3.12 -y || true
source /root/miniconda3/bin/activate info_depreciation

# RTX 5090 (Blackwell) 需要 GCC 11+ 编译器
export TORCH_CUDA_ARCH_LIST="10.0"
ok "环境就绪"

info "步骤 2/6: 安装 PyTorch 2.7.0 (CUDA 12.8)"
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
ok "PyTorch安装完成"

info "步骤 3/6: 安装核心依赖"
# transformers 4.52+ 与 vLLM 0.9 配合更稳定
pip install transformers==4.52.3
pip install tokenizers==0.21.0
pip install datasets==2.21.0
pip install accelerate==0.34.0
pip install bitsandbytes==0.44.0
ok "核心依赖完成"

info "步骤 4/6: 安装 vLLM 0.9.2 (比0.9.0更稳定)"
# vLLM 0.9.0 有已知bug(numerical error, FusedMoE)，社区推荐0.9.2
pip install vllm==0.9.2
# 验证Blackwell支持
python -c "import torch; cap=torch.cuda.get_device_capability(0); print(f'Compute Capability: {cap[0]}.{cap[1]}')"
# vLLM 0.9 默认启用V1引擎不稳定，强制回退V0
export VLLM_USE_V1=0
if ! grep -q "VLLM_USE_V1" ~/.bashrc; then
    echo 'export VLLM_USE_V1=0' >> ~/.bashrc
fi
ok "vLLM安装完成(V0引擎)"

info "步骤 5/6: 安装辅助依赖"
pip install scipy "numpy<2" tqdm pandas matplotlib statsmodels
pip install huggingface-hub
ok "辅助依赖完成"

info "步骤 6/6: 配置HF镜像"
mkdir -p ~/.config/huggingface
cat > ~/.config/huggingface/config.yaml << 'EOF'
hf_endpoint: https://hf-mirror.com
EOF
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface_cache
if ! grep -q "HF_ENDPOINT" ~/.bashrc; then
    echo '' >> ~/.bashrc
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
    echo 'export HF_HOME=/root/autodl-tmp/huggingface_cache' >> ~/.bashrc
fi
mkdir -p /root/autodl-tmp/huggingface_cache
ok "配置完成"

python << 'PYEOF'
import torch, transformers, vllm
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"vLLM: {vllm.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PYEOF

ok "全部完成！"
echo ""
echo "========================================"
echo -e "${GREEN}  RTX 5090 + vLLM 0.9 环境配置完成${NC}"
echo "========================================"
echo ""
echo "  运行: python run_real_experiments.py --experiment all --model_size llama3_8b --use_vllm"
echo "========================================"
