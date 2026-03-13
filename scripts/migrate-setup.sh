#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
info() { echo -e "${BLUE}[i]${NC} $*"; }

echo "============================================"
echo "  LLM 项目迁移 - 新机器环境配置"
echo "============================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIGRATION_DIR="${SCRIPT_DIR}"
TARGET_DIR=""

if [[ -f "${MIGRATION_DIR}/repo.bundle" ]]; then
    info "检测到迁移包: ${MIGRATION_DIR}"
else
    warn "当前目录不是迁移包目录，将按照已有项目目录处理。"
fi

# ── 检测目标项目目录 ────────────────────────────────
if [[ -n "${1:-}" ]]; then
    TARGET_DIR="$1"
elif [[ -f "${MIGRATION_DIR}/repo.bundle" ]]; then
    read -p "输入项目安装路径 (默认: ${HOME}/llm): " TARGET_DIR
    TARGET_DIR="${TARGET_DIR:-${HOME}/llm}"
else
    TARGET_DIR="$(dirname "${SCRIPT_DIR}")"
fi
info "项目目标路径: ${TARGET_DIR}"

# ── 1. 安装系统依赖 ─────────────────────────────────
echo ""
echo "── 步骤 1/7: 系统依赖 ──────────────────────"

check_cmd() {
    command -v "$1" &>/dev/null
}

NEED_APT=()
check_cmd gcc || NEED_APT+=(build-essential)
check_cmd cmake || NEED_APT+=(cmake)
check_cmd git || NEED_APT+=(git)
check_cmd curl || NEED_APT+=(curl)
check_cmd wget || NEED_APT+=(wget)

if [[ ${#NEED_APT[@]} -gt 0 ]]; then
    info "安装系统包: ${NEED_APT[*]}"
    sudo apt update
    sudo apt install -y "${NEED_APT[@]}"
    log "系统包安装完成"
else
    log "系统依赖已满足"
fi

# ── 2. 检查 NVIDIA 驱动和 CUDA ──────────────────────
echo ""
echo "── 步骤 2/7: NVIDIA CUDA ──────────────────────"

if check_cmd nvidia-smi; then
    log "NVIDIA 驱动已安装"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    err "未检测到 NVIDIA 驱动！"
    echo "    请先安装 NVIDIA 驱动: https://www.nvidia.com/Download/index.aspx"
    echo "    或: sudo apt install nvidia-driver-565"
    exit 1
fi

if check_cmd nvcc; then
    log "CUDA Toolkit 已安装: $(nvcc --version | tail -1)"
else
    warn "CUDA Toolkit 未安装"
    echo "    请安装 CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    read -p "是否继续？部分功能可能不可用 (y/n): " CONT
    [[ "${CONT}" != "y" ]] && exit 1
fi

if ldconfig -p 2>/dev/null | grep -q libnccl; then
    log "NCCL 已安装"
else
    warn "NCCL 未安装，尝试安装..."
    sudo apt install -y libnccl2 libnccl-dev 2>/dev/null || {
        warn "NCCL 自动安装失败，请手动安装"
        echo "    https://developer.nvidia.com/nccl"
    }
fi

# ── 3. 安装 xmake ──────────────────────────────────
echo ""
echo "── 步骤 3/7: xmake ──────────────────────"

if check_cmd xmake; then
    log "xmake 已安装: $(xmake --version 2>&1 | head -1)"
else
    info "安装 xmake..."
    curl -fsSL https://xmake.io/shget.text | bash
    export PATH="${HOME}/.local/bin:${PATH}"
    source "${HOME}/.xmake/profile" 2>/dev/null || true
    if check_cmd xmake; then
        log "xmake 安装完成"
    else
        err "xmake 安装失败，请手动安装: https://xmake.io/#/getting_started"
        exit 1
    fi
fi

# ── 4. Python ──────────────────────────────────────
echo ""
echo "── 步骤 4/7: Python 环境 ──────────────────────"

PYTHON_CMD=""
for cmd in python3.13 python3 python; do
    if check_cmd "${cmd}"; then
        PY_VER=$("${cmd}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PY_MAJOR=$(echo "${PY_VER}" | cut -d. -f1)
        PY_MINOR=$(echo "${PY_VER}" | cut -d. -f2)
        if [[ ${PY_MAJOR} -ge 3 && ${PY_MINOR} -ge 10 ]]; then
            PYTHON_CMD="${cmd}"
            break
        fi
    fi
done

if [[ -n "${PYTHON_CMD}" ]]; then
    log "Python 已安装: $(${PYTHON_CMD} --version)"
else
    warn "需要 Python 3.10+，尝试安装 Python 3.13..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt install -y python3.13 python3.13-venv python3.13-dev
    PYTHON_CMD="python3.13"
    log "Python 3.13 安装完成"
fi

if ! check_cmd pip && ! check_cmd pip3; then
    sudo apt install -y python3-pip
fi

# ── 5. Node.js ─────────────────────────────────────
echo ""
echo "── 步骤 5/7: Node.js 环境 ──────────────────────"

if check_cmd node; then
    NODE_VER=$(node --version | sed 's/v//' | cut -d. -f1)
    if [[ ${NODE_VER} -ge 20 ]]; then
        log "Node.js 已安装: $(node --version)"
    else
        warn "Node.js 版本过低 ($(node --version))，需要 v20+"
        info "安装 Node.js 24..."
        curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
        sudo apt install -y nodejs
        log "Node.js 更新完成: $(node --version)"
    fi
else
    info "安装 Node.js 24..."
    curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
    sudo apt install -y nodejs
    log "Node.js 安装完成: $(node --version)"
fi

# ── 6. 恢复项目 ────────────────────────────────────
echo ""
echo "── 步骤 6/7: 恢复项目文件 ──────────────────────"

if [[ -f "${MIGRATION_DIR}/repo.bundle" && ! -d "${TARGET_DIR}/.git" ]]; then
    info "从 bundle 恢复 Git 仓库..."
    git clone "${MIGRATION_DIR}/repo.bundle" "${TARGET_DIR}"
    cd "${TARGET_DIR}"
    git remote set-url origin https://github.com/wxhwqy/llm.git
    log "代码恢复完成"
elif [[ -d "${TARGET_DIR}/.git" ]]; then
    log "项目目录已存在: ${TARGET_DIR}"
    cd "${TARGET_DIR}"
else
    info "从 GitHub 克隆..."
    git clone https://github.com/wxhwqy/llm.git "${TARGET_DIR}"
    cd "${TARGET_DIR}"
    log "克隆完成"
fi

if [[ -d "${MIGRATION_DIR}/models" ]]; then
    info "恢复模型权重..."
    mkdir -p "${TARGET_DIR}/llm_service/models"
    for model_dir in "${MIGRATION_DIR}/models"/*/; do
        if [[ -d "${model_dir}" ]]; then
            model_name=$(basename "${model_dir}")
            if [[ -d "${TARGET_DIR}/llm_service/models/${model_name}" ]]; then
                warn "模型 ${model_name} 已存在，跳过"
            else
                info "恢复 ${model_name}..."
                cp -r "${model_dir}" "${TARGET_DIR}/llm_service/models/${model_name}"
                log "${model_name} 恢复完成"
            fi
        fi
    done
fi

if [[ -f "${MIGRATION_DIR}/configs/web.env" ]]; then
    cp "${MIGRATION_DIR}/configs/web.env" "${TARGET_DIR}/web/.env"
    log "web/.env 恢复完成"
    warn "请检查并按需修改 web/.env 中的配置"
fi

if [[ -f "${MIGRATION_DIR}/configs/dev.db" ]]; then
    cp "${MIGRATION_DIR}/configs/dev.db" "${TARGET_DIR}/web/dev.db"
    log "dev.db 恢复完成"
fi

# ── 7. 构建与安装 ──────────────────────────────────
echo ""
echo "── 步骤 7/7: 构建项目 ──────────────────────"

cd "${TARGET_DIR}"

info "构建 C++ 推理引擎..."
cd llm_service
if check_cmd nvcc; then
    xmake f --nv-gpu=y -m release -y
else
    warn "CUDA 不可用，仅构建 CPU 版本"
    xmake f -m release -y
fi
xmake build -j"$(nproc)"
log "C++ 构建完成"

info "安装 Python 依赖..."
cd python
pip install -e . 2>/dev/null || pip3 install -e .
cd ../api
pip install -r requirements.txt 2>/dev/null || pip3 install -r requirements.txt
log "Python 依赖安装完成"
cd "${TARGET_DIR}"

info "安装前端依赖..."
cd web
npm install
npx prisma generate
log "前端依赖安装完成"

if [[ ! -f dev.db ]]; then
    info "初始化数据库..."
    npx prisma migrate dev --name init
    npm run db:seed 2>/dev/null || warn "seed 失败，可手动运行: npm run db:seed"
fi

cd "${TARGET_DIR}"

echo ""
echo "============================================"
echo "  环境配置完成！"
echo "============================================"
echo ""
info "项目路径: ${TARGET_DIR}"
echo ""
info "启动 LLM 服务:"
echo "    cd ${TARGET_DIR}/llm_service && python -m api.main"
echo ""
info "启动前端:"
echo "    cd ${TARGET_DIR}/web && npm run dev"
echo ""
info "运行验证脚本:"
echo "    bash ${TARGET_DIR}/scripts/migrate-verify.sh"
echo ""
