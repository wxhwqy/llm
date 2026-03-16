#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
EXPORT_DIR="${PROJECT_DIR}/llm-migration"

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
echo "  LLM 项目迁移 - 源机器导出"
echo "============================================"
echo ""

if [[ ! -d "${PROJECT_DIR}/.git" ]]; then
    err "未找到 Git 仓库: ${PROJECT_DIR}"
    exit 1
fi

info "项目目录: ${PROJECT_DIR}"
info "导出目录: ${EXPORT_DIR}"
echo ""

rm -rf "${EXPORT_DIR}"
mkdir -p "${EXPORT_DIR}/configs" "${EXPORT_DIR}/models"

# ── 1. 记录环境信息 ─────────────────────────────────
info "记录当前环境信息..."
cat > "${EXPORT_DIR}/env-snapshot.txt" <<ENVEOF
=== 环境快照 $(date '+%Y-%m-%d %H:%M:%S') ===

OS:        $(uname -s -r -m)
Python:    $(python3 --version 2>&1 || echo "未安装")
Node.js:   $(node --version 2>&1 || echo "未安装")
npm:       $(npm --version 2>&1 || echo "未安装")
xmake:     $(xmake --version 2>&1 | head -1 || echo "未安装")
CUDA:      $(nvcc --version 2>&1 | tail -1 || echo "未安装")
GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || echo "无GPU")

pip packages:
$(pip list 2>/dev/null | grep -E "torch|transformers|fastapi|uvicorn|safetensors|numpy" || echo "  无法获取")
ENVEOF
log "环境快照已保存"

# ── 2. 导出 Git 仓库 ─────────────────────────────────
info "导出 Git 仓库 (bundle 格式)..."
cd "${PROJECT_DIR}"
git bundle create "${EXPORT_DIR}/repo.bundle" --all
BUNDLE_SIZE=$(du -sh "${EXPORT_DIR}/repo.bundle" | cut -f1)
log "Git bundle 完成 (${BUNDLE_SIZE})"

# ── 3. 备份配置文件 ─────────────────────────────────
info "备份配置文件..."

if [[ -f "${PROJECT_DIR}/web/.env" ]]; then
    cp "${PROJECT_DIR}/web/.env" "${EXPORT_DIR}/configs/web.env"
    log "web/.env 已备份"
else
    warn "web/.env 不存在，跳过"
fi

if [[ -f "${PROJECT_DIR}/llm_service/models.json" ]]; then
    cp "${PROJECT_DIR}/llm_service/models.json" "${EXPORT_DIR}/configs/models.json"
    log "models.json 已备份"
fi

# ── 4. 备份数据库 ─────────────────────────────────────
if [[ -f "${PROJECT_DIR}/web/dev.db" ]]; then
    cp "${PROJECT_DIR}/web/dev.db" "${EXPORT_DIR}/configs/dev.db"
    DB_SIZE=$(du -sh "${PROJECT_DIR}/web/dev.db" | cut -f1)
    log "dev.db 已备份 (${DB_SIZE})"
else
    warn "dev.db 不存在，跳过"
fi

# ── 5. 打包模型权重 ─────────────────────────────────
MODELS_DIR="${PROJECT_DIR}/llm_service/models"
if [[ -d "${MODELS_DIR}" ]]; then
    echo ""
    info "检测到以下模型:"
    for model_dir in "${MODELS_DIR}"/*/; do
        if [[ -d "${model_dir}" ]]; then
            model_name=$(basename "${model_dir}")
            model_size=$(du -sh "${model_dir}" | cut -f1)
            echo "    ${model_name}  (${model_size})"
        fi
    done
    echo ""

    read -p "是否复制模型权重到导出目录？(y/n, 默认 y): " COPY_MODELS
    COPY_MODELS=${COPY_MODELS:-y}

    if [[ "${COPY_MODELS}" == "y" || "${COPY_MODELS}" == "Y" ]]; then
        info "选择要迁移的模型 (输入序号，多个用空格分隔，输入 all 全选):"
        MODELS=()
        i=1
        for model_dir in "${MODELS_DIR}"/*/; do
            if [[ -d "${model_dir}" ]]; then
                model_name=$(basename "${model_dir}")
                echo "    ${i}) ${model_name}"
                MODELS+=("${model_name}")
                ((i++))
            fi
        done

        read -p "选择: " MODEL_CHOICE
        SELECTED=()

        if [[ "${MODEL_CHOICE}" == "all" ]]; then
            SELECTED=("${MODELS[@]}")
        else
            for idx in ${MODEL_CHOICE}; do
                if [[ ${idx} -ge 1 && ${idx} -le ${#MODELS[@]} ]]; then
                    SELECTED+=("${MODELS[$((idx-1))]}")
                fi
            done
        fi

        for model in "${SELECTED[@]}"; do
            info "复制模型 ${model} ..."
            cp -r "${MODELS_DIR}/${model}" "${EXPORT_DIR}/models/${model}"
            log "${model} 复制完成"
        done
    else
        warn "跳过模型复制。你需要手动传输模型到新机器。"
        info "提示: 可以用 rsync 直接传输模型目录:"
        echo "    rsync -avhP ${MODELS_DIR}/ user@新机器:/path/to/llm/llm_service/models/"
    fi
else
    warn "未找到模型目录: ${MODELS_DIR}"
fi

# ── 6. 生成传输脚本 ─────────────────────────────────
cat > "${EXPORT_DIR}/rsync-to-remote.sh" <<'RSYNCEOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "用法: $0 user@host [远程目录]"
    echo "例: $0 lma@192.168.1.100 /home/lma/llm-migration"
    exit 1
fi

REMOTE="$1"
REMOTE_DIR="${2:-~/llm-migration}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "传输到 ${REMOTE}:${REMOTE_DIR} ..."
rsync -avhP --compress "${SCRIPT_DIR}/" "${REMOTE}:${REMOTE_DIR}/"
echo "传输完成！"
echo "在远程机器上执行: cd ${REMOTE_DIR} && bash migrate-setup.sh"
RSYNCEOF
chmod +x "${EXPORT_DIR}/rsync-to-remote.sh"

# ── 7. 复制安装脚本到导出目录 ─────────────────────────
for script in migrate-setup.sh migrate-verify.sh; do
    if [[ -f "${SCRIPT_DIR}/${script}" ]]; then
        cp "${SCRIPT_DIR}/${script}" "${EXPORT_DIR}/${script}"
        chmod +x "${EXPORT_DIR}/${script}"
    fi
done

# ── 汇总 ────────────────────────────────────────────
echo ""
echo "============================================"
echo "  导出完成！"
echo "============================================"
echo ""
TOTAL_SIZE=$(du -sh "${EXPORT_DIR}" | cut -f1)
info "导出目录: ${EXPORT_DIR}"
info "总大小: ${TOTAL_SIZE}"
echo ""
info "目录结构:"
find "${EXPORT_DIR}" -maxdepth 2 -not -path '*/models/*/*.safetensors' | head -30 | sed "s|${EXPORT_DIR}/|  |"
echo ""
info "下一步:"
echo "  方式1: bash ${EXPORT_DIR}/rsync-to-remote.sh user@新机器IP"
echo "  方式2: 复制 ${EXPORT_DIR} 到移动硬盘"
echo ""
