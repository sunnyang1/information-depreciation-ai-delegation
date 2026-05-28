#!/bin/bash
# setup_env_v3_vllm.sh - AutoDL环境配置 v3.0 (with vLLM acceleration)
# 推荐GPU: RTX 4090 / A100 (CUDA 12.1+)
# 版本组合(2026验证): PyTorch 2.5.1 + transformers 4.48.1 + vLLM 0.7.0

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

# ============ 0. 环境检查 ============
info "步骤 0/7: 检查CUDA版本"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    info "检测到CUDA版本: $CUDA_VERSION"
    if [[ "$CUDA_VERSION" != "12.1" && "$CUDA_VERSION" != "12.4" && "$CUDA_VERSION" != "12.2" ]]; then
        warn "推荐CUDA 12.1或12.4，当前是$CUDA_VERSION"
        warn "vLLM 0.7.0可能需要cu121，继续安装但可能有问题"
    fi
else
    warn "未检测到nvcc"
fi

# ============ 1. 创建conda环境 ============
info "步骤 1/7: 创建conda环境 'info_depreciation' (Python 3.10)"
conda create -n info_depreciation python=3.10 -y || true
source /root/miniconda3/bin/activate info_depreciation
ok "conda环境就绪"

# ============ 2. 安装PyTorch 2.5.1 (CUDA 12.1) ============
info "步骤 2/7: 安装 PyTorch 2.5.1 (CUDA 12.1)"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
ok "PyTorch安装完成"

# ============ 3. 安装核心依赖 ============
info "步骤 3/7: 安装transformers及相关库"
pip install transformers==4.48.0
pip install tokenizers==0.21.0
pip install datasets==2.21.0
pip install accelerate==0.34.0
pip install bitsandbytes==0.44.0
ok "核心依赖安装完成"

# ============ 4. 安装vLLM ============
info "步骤 4/7: 安装vLLM 0.7.0 (推理加速)"
pip install vllm==0.7.0 || {
    warn "vLLM 0.7.0安装失败，尝试0.6.3版本"
    pip install vllm==0.6.3
}
ok "vLLM安装完成"

# ============ 5. 安装其他依赖 ============
info "步骤 5/7: 安装统计/可视化依赖"
pip install scipy "numpy<2" tqdm pandas matplotlib statsmodels
pip install huggingface-hub
ok "辅助依赖安装完成"

# ============ 6. 配置HF镜像 ============
info "步骤 6/7: 配置HuggingFace国内镜像"
mkdir -p ~/.config/huggingface
cat > ~/.config/huggingface/config.yaml << 'EOF'
hf_endpoint: https://hf-mirror.com
EOF
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface_cache
if ! grep -q "HF_ENDPOINT" ~/.bashrc; then
    echo '' >> ~/.bashrc
    echo '# HuggingFace镜像配置' >> ~/.bashrc
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
    echo 'export HF_HOME=/root/autodl-tmp/huggingface_cache' >> ~/.bashrc
fi
mkdir -p /root/autodl-tmp/huggingface_cache
ok "HF镜像配置完成"

# ============ 7. 验证 ============
info "步骤 7/7: 验证关键包"
python << 'PYEOF'
import sys
errors = []

def check(pkg, name):
    try:
        __import__(pkg)
        print(f"  [OK] {name}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {name}: {e}")
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
    if not check(mod, name):
        errors.append(name)

for mod, name in optional:
    if check(mod, name + " (optional)"):
        pass
    else:
        print(f"  [WARN] {name} (optional)")

if errors:
    print(f"\n[ERROR] 关键包未安装: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\n[OK] 所有关键包验证通过")
PYEOF

# 显示版本信息
python -c "
import torch, transformers, vllm, datasets
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'vLLM: {vllm.__version__}')
print(f'Datasets: {datasets.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

ok "环境配置全部完成！"
echo ""
echo "========================================"
echo -e "${GREEN}  v3.0 vLLM环境配置完成${NC}"
echo "========================================"
echo ""
echo "  激活: source /root/miniconda3/bin/activate info_depreciation"
echo ""
echo "  支持模型(公开,无需token):"
echo "    - meta-llama/Llama-3.1-8B-Instruct"
echo "    - mistralai/Mistral-7B-Instruct-v0.3"
echo "    - Qwen/Qwen2.5-7B-Instruct"
echo "    - google/gemma-2-9b-it"
echo ""
echo "  运行实验:"
echo "    python run_real_experiments.py --experiment all --model_size 8b-instruct --use_vllm"
echo "========================================"
