#!/bin/bash
# setup_env_v3_vllm28.sh - PyTorch 2.8.0 + CUDA 12.8 + vLLM 0.16.0 + Transformers 5.0
# 推荐GPU: RTX 5090 (Blackwell)
# 注意: 这是前沿版本组合，PyTorch 2.8 + Transformers 5.0 可能有 breaking changes

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

# RTX 5090 (Blackwell) 设置
export TORCH_CUDA_ARCH_LIST="10.0"
ok "环境就绪"

info "步骤 2/6: 安装 PyTorch 2.8.0 (CUDA 12.8)"
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
ok "PyTorch安装完成"

info "步骤 3/6: 安装核心依赖 (Transformers 5.x)"
pip install transformers==5.0.0
pip install tokenizers==0.21.0
pip install datasets==2.21.0
pip install accelerate==0.34.0
pip install bitsandbytes==0.44.0
ok "核心依赖完成"

info "步骤 4/6: 安装 vLLM 0.16.0 (支持PyTorch 2.8)"
pip install vllm==0.16.0 || {
    warn "vLLM 0.16.0安装失败，尝试最新稳定版"
    pip install vllm
}
ok "vLLM安装完成"

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

# 验证
python << 'PYEOF'
import torch, transformers, vllm
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"vLLM: {vllm.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"CC: {torch.cuda.get_device_capability(0)}")
PYEOF

ok "全部完成！"
echo ""
echo "========================================"
echo -e "${GREEN}  PyTorch 2.8 + vLLM 0.16 环境配置完成${NC}"
echo "========================================"
echo ""
echo "  注意: Transformers 5.0 可能有API变化"
echo "  如遇报错请贴日志给我修复"
echo ""
echo "  运行: python run_real_experiments.py --experiment all --model_size llama3_8b --use_vllm"
echo "========================================"
