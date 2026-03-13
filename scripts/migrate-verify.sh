#!/usr/bin/env bash
set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

pass() { echo -e "  ${GREEN}✓${NC} $*"; ((PASS++)); }
fail() { echo -e "  ${RED}✗${NC} $*"; ((FAIL++)); }
skip() { echo -e "  ${YELLOW}!${NC} $*"; ((WARN++)); }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  LLM 项目迁移 - 验证检查"
echo "============================================"
echo ""

# ── 系统工具 ────────────────────────────────────────
echo "── 系统工具 ──"

for cmd in git gcc g++ cmake curl; do
    if command -v "${cmd}" &>/dev/null; then
        pass "${cmd}: $(${cmd} --version 2>&1 | head -1)"
    else
        fail "${cmd} 未安装"
    fi
done

# ── GPU / CUDA ──────────────────────────────────────
echo ""
echo "── GPU / CUDA ──"

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)
    pass "NVIDIA 驱动: ${GPU_INFO}"
else
    fail "nvidia-smi 不可用"
fi

if command -v nvcc &>/dev/null; then
    pass "CUDA: $(nvcc --version 2>&1 | tail -1)"
else
    fail "nvcc 不可用 (CUDA Toolkit)"
fi

if ldconfig -p 2>/dev/null | grep -q libnccl; then
    pass "NCCL 已安装"
else
    skip "NCCL 未检测到（多卡并行需要）"
fi

# ── 开发工具 ────────────────────────────────────────
echo ""
echo "── 开发工具 ──"

if command -v xmake &>/dev/null; then
    pass "xmake: $(xmake --version 2>&1 | head -1 | sed 's/\x1b\[[0-9;]*m//g')"
else
    fail "xmake 未安装"
fi

if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1)
    pass "Python: ${PY_VER}"
else
    fail "Python3 未安装"
fi

if command -v node &>/dev/null; then
    pass "Node.js: $(node --version)"
else
    fail "Node.js 未安装"
fi

if command -v npm &>/dev/null; then
    pass "npm: $(npm --version)"
else
    fail "npm 未安装"
fi

# ── 项目文件 ────────────────────────────────────────
echo ""
echo "── 项目文件 ──"

cd "${PROJECT_DIR}"

if [[ -d .git ]]; then
    BRANCH=$(git branch --show-current)
    pass "Git 仓库: 分支 ${BRANCH}"
else
    fail "Git 仓库不存在"
fi

for f in llm_service/xmake.lua llm_service/models.json llm_service/api/main.py web/package.json web/prisma/schema.prisma; do
    if [[ -f "${f}" ]]; then
        pass "文件存在: ${f}"
    else
        fail "文件缺失: ${f}"
    fi
done

if [[ -f web/.env ]]; then
    pass "web/.env 存在"
else
    fail "web/.env 缺失（需要手动创建）"
fi

# ── 模型权重 ────────────────────────────────────────
echo ""
echo "── 模型权重 ──"

MODELS_DIR="${PROJECT_DIR}/llm_service/models"
if [[ -d "${MODELS_DIR}" ]]; then
    MODEL_COUNT=0
    for model_dir in "${MODELS_DIR}"/*/; do
        if [[ -d "${model_dir}" ]]; then
            model_name=$(basename "${model_dir}")
            model_size=$(du -sh "${model_dir}" 2>/dev/null | cut -f1)
            if ls "${model_dir}"/*.safetensors &>/dev/null; then
                pass "模型 ${model_name} (${model_size})"
                ((MODEL_COUNT++))
            else
                fail "模型 ${model_name} 不完整 (缺少 .safetensors)"
            fi
        fi
    done
    if [[ ${MODEL_COUNT} -eq 0 ]]; then
        skip "未找到模型权重文件"
    fi
else
    skip "模型目录不存在: ${MODELS_DIR}"
fi

# ── Python 依赖 ─────────────────────────────────────
echo ""
echo "── Python 依赖 ──"

for pkg in torch transformers safetensors fastapi uvicorn numpy; do
    if python3 -c "import ${pkg}" &>/dev/null; then
        VER=$(python3 -c "import ${pkg}; print(${pkg}.__version__)" 2>/dev/null || echo "?")
        pass "${pkg} ${VER}"
    else
        fail "${pkg} 未安装"
    fi
done

# ── llaisys 引擎 ────────────────────────────────────
echo ""
echo "── llaisys 推理引擎 ──"

if python3 -c "import llaisys" &>/dev/null; then
    pass "llaisys Python 绑定可用"
else
    fail "llaisys 不可用（需要构建: cd llm_service && xmake build）"
fi

SO_FILE=$(find "${PROJECT_DIR}/llm_service/python/llaisys/libllaisys/" -name "*.so" 2>/dev/null | head -1)
if [[ -n "${SO_FILE}" ]]; then
    pass "共享库: $(basename "${SO_FILE}")"
else
    fail "libllaisys.so 不存在（需要构建 C++ 引擎）"
fi

# ── PyTorch GPU ─────────────────────────────────────
echo ""
echo "── PyTorch GPU 支持 ──"

if python3 -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    pass "PyTorch CUDA 可用, GPU 数量: ${GPU_COUNT}"
else
    fail "PyTorch CUDA 不可用"
fi

# ── 前端 ────────────────────────────────────────────
echo ""
echo "── 前端 (web/) ──"

if [[ -d web/node_modules ]]; then
    pass "node_modules 已安装"
else
    fail "node_modules 缺失 (需要运行: cd web && npm install)"
fi

if [[ -d web/src/generated ]]; then
    pass "Prisma client 已生成"
else
    fail "Prisma client 未生成 (需要运行: cd web && npx prisma generate)"
fi

if [[ -f web/dev.db ]]; then
    DB_SIZE=$(du -sh web/dev.db | cut -f1)
    pass "数据库 dev.db (${DB_SIZE})"
else
    skip "dev.db 不存在 (可运行: cd web && npx prisma migrate dev)"
fi

# ── 汇总 ────────────────────────────────────────────
echo ""
echo "============================================"
echo -e "  结果: ${GREEN}${PASS} 通过${NC}  ${RED}${FAIL} 失败${NC}  ${YELLOW}${WARN} 警告${NC}"
echo "============================================"

if [[ ${FAIL} -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}迁移验证全部通过！可以正常使用。${NC}"
    echo ""
    echo "启动命令:"
    echo "  LLM 服务: cd ${PROJECT_DIR}/llm_service && python -m api.main"
    echo "  前端:     cd ${PROJECT_DIR}/web && npm run dev"
else
    echo ""
    echo -e "${RED}有 ${FAIL} 项检查未通过，请按上述提示修复。${NC}"
fi

exit ${FAIL}
