# llaisys LLM Service

基于自研 C++ 推理引擎的大模型服务，提供 OpenAI 兼容 API。

## 支持的模型

| 模型 | 类型 | 量化 | 设备 |
|---|---|---|---|
| Qwen3 (8B/14B/32B) | Dense Transformer | FP8 | GPU (支持TP多卡) |
| Qwen3.5-9B | Hybrid DeltaNet+Attn | BF16/FP8 | GPU |
| Qwen3.5-35B-A3B | MoE (256 experts) | GPTQ INT4 | CPU |

## 环境要求

- **操作系统**: Linux (推荐) / macOS
- **编译工具**: [xmake](https://xmake.io) >= 2.8
- **C++ 编译器**: GCC 11+ 或 Clang 15+ (需支持 C++17)
- **Python**: 3.10+
- **GPU (可选)**: CUDA 12.0+, cuBLAS, NCCL

Python 依赖：
```
torch
safetensors
transformers
fastapi
uvicorn
```

## 编译

```bash
cd llm_service

# CPU-only 编译
xmake -y

# 启用 GPU 支持
xmake -y -o nv-gpu
```

编译完成后共享库会自动拷贝到 `python/llaisys/libllaisys/`。

## 下载模型

使用 HuggingFace CLI 下载模型权重到 `llm_service/models/` 目录：

```bash
# 示例：下载 Qwen3.5-35B MoE GPTQ 量化模型（CPU 推理，~24GB）
huggingface-cli download Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --local-dir llm_service/models/qwen3_5_35b_a3b_gptq_int4

# 示例：下载 Qwen3-32B FP8（GPU 推理）
huggingface-cli download Qwen/Qwen3-32B-FP8 \
  --local-dir llm_service/models/qwen3_32b_fp8
```

## 启动服务

### 方式一：FastAPI 服务（推荐）

1. 配置 `llm_service/models.json`：

```json
[
  {
    "id": "qwen3_5-moe",
    "name": "Qwen3.5 35B MoE",
    "path": "models/qwen3_5_35b_a3b_gptq_int4",
    "model_type": "qwen3_5_moe",
    "max_seq_len": 8192,
    "device": "cpu",
    "device_ids": [],
    "tp_size": 1
  }
]
```

支持的 `model_type`：`qwen3`、`qwen3_5`、`qwen3_5_moe`

2. 启动服务：

```bash
cd llm_service
LLM_DEFAULT_MODEL=qwen3_5-moe uvicorn api.main:app --host 0.0.0.0 --port 8000
```

3. 调用 API（兼容 OpenAI 格式）：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5-moe",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.6,
    "max_tokens": 512
  }'
```

### 方式二：Python 直接调用

```python
from llaisys.models import Qwen3_5Moe  # 或 Qwen3_5, Qwen3

model = Qwen3_5Moe("models/qwen3_5_35b_a3b_gptq_int4", device="cpu")

for token_text in model.stream_generate("你好", max_new_tokens=256):
    print(token_text, end="", flush=True)
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_HOST` | `0.0.0.0` | 监听地址 |
| `LLM_PORT` | `8000` | 监听端口 |
| `LLM_DEFAULT_MODEL` | `qwen3-32b` | 默认模型 ID |
| `LLM_MODELS_CONFIG` | `models.json` | 模型配置文件路径 |
| `LLM_DEVICE` | `nvidia` | 设备类型 (`nvidia` / `cpu`) |
| `LLM_DEVICE_IDS` | `[0,1]` | GPU 设备 ID 列表 |
| `LLM_TP_SIZE` | `2` | 张量并行度 |
| `LLM_MAX_SEQ_LEN` | `8192` | 最大序列长度 |
| `LLM_DEFAULT_TEMPERATURE` | `0.6` | 默认采样温度 |
| `LLM_MAX_CONCURRENT` | `1` | 最大并发推理数 |

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat/completions` | 聊天补全（支持流式） |
| GET | `/v1/models` | 列出可用模型 |
| GET | `/health` | 健康检查 |
