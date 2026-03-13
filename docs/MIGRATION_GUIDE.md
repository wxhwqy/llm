# LLM 项目迁移指南

本文档指导你将 LLM 项目从当前机器迁移到新机器。

## 当前环境快照

| 项目 | 值 |
|------|-----|
| OS | Linux x86_64 |
| CPU | AMD EPYC 7543 32-Core |
| GPU | 2× NVIDIA RTX 4090 (24GB) |
| RAM | 125GB |
| Python | 3.13.5 |
| Node.js | v24.14.0 |
| xmake | v3.0.7 |
| CUDA | /usr/local/cuda |

## 项目组成

```
llm/
├── llm_service/        # C++/CUDA 推理引擎 + Python API
│   ├── src/            # C++17 源码 (CUDA kernels, 模型实现)
│   ├── python/         # llaisys Python 绑定
│   ├── api/            # FastAPI REST 服务
│   ├── models/         # ⚠️ 模型权重 (~57GB, 不在 git 中)
│   └── xmake.lua       # 构建脚本
├── web/                # Next.js 前端
│   ├── src/            # App Router + API Routes
│   ├── prisma/         # 数据库 schema
│   └── dev.db          # SQLite 数据库
└── docs/
```

## 迁移步骤概览

1. **在源机器上打包** — 运行 `scripts/migrate-export.sh`
2. **传输文件到新机器** — rsync / scp / 移动硬盘
3. **在新机器上安装依赖** — 运行 `scripts/migrate-setup.sh`
4. **验证迁移结果** — 运行 `scripts/migrate-verify.sh`

---

## 第一步：在源机器上打包

```bash
cd /home/lma/llm
bash scripts/migrate-export.sh
```

该脚本会：
- 导出 Git 仓库（包含所有提交历史）
- 打包模型权重（约 57GB）
- 备份环境配置文件（`.env`、`models.json`）
- 备份数据库（`dev.db`）
- 记录当前环境信息

产出文件保存在 `llm-migration/` 目录中。

### 文件传输

根据你的网络情况选择传输方式：

**方式 A：局域网 rsync（推荐，支持断点续传）**

```bash
rsync -avhP --compress llm-migration/ user@新机器IP:/home/user/llm-migration/
```

**方式 B：移动硬盘**

```bash
cp -r llm-migration/ /mnt/外接硬盘/llm-migration/
```

**方式 C：仅传输模型（代码走 git clone）**

```bash
# 新机器上先 clone 代码
git clone https://github.com/wxhwqy/llm.git

# 然后只传模型和配置
rsync -avhP llm-migration/models/ user@新机器IP:/home/user/llm/llm_service/models/
scp llm-migration/configs/* user@新机器IP:/home/user/llm-migration/configs/
```

---

## 第二步：在新机器上安装依赖

### 硬件要求

| 组件 | 最低要求 | 推荐 |
|------|----------|------|
| GPU | 1× NVIDIA GPU (≥16GB VRAM) | 2× RTX 4090 |
| RAM | 32GB | 64GB+ |
| 磁盘 | 100GB 可用空间 | 200GB+ SSD |
| CUDA Compute | ≥ 7.0 (Volta) | ≥ 8.9 (Ada) |

### 自动安装

```bash
cd /home/user/llm-migration  # 或者你的迁移目录
bash migrate-setup.sh
```

### 手动安装（如果自动脚本不适用）

#### 2.1 系统工具

```bash
sudo apt update && sudo apt install -y \
    build-essential cmake git curl wget \
    libssl-dev pkg-config
```

#### 2.2 CUDA Toolkit

访问 https://developer.nvidia.com/cuda-downloads 下载安装。
确保 `nvcc --version` 可用且 `/usr/local/cuda` 存在。

#### 2.3 NCCL

```bash
sudo apt install -y libnccl2 libnccl-dev
```

#### 2.4 xmake

```bash
curl -fsSL https://xmake.io/shget.text | bash
source ~/.xmake/profile
```

#### 2.5 Python 3.13

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.13 python3.13-venv python3.13-dev python3-pip
```

#### 2.6 Node.js 24

```bash
curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
sudo apt install -y nodejs
```

---

## 第三步：恢复项目

### 3.1 恢复代码

**从 git bundle 恢复：**

```bash
git clone llm-migration/repo.bundle llm
cd llm
git remote set-url origin https://github.com/wxhwqy/llm.git
```

**或直接从 GitHub 克隆：**

```bash
git clone https://github.com/wxhwqy/llm.git
cd llm
```

### 3.2 恢复模型权重

```bash
cp -r llm-migration/models/* llm_service/models/
```

或者如果模型还在原始下载源，重新下载也可以。

### 3.3 恢复配置文件

```bash
cp llm-migration/configs/web.env web/.env
cp llm-migration/configs/dev.db web/dev.db
```

**重要**：编辑 `web/.env`，根据新环境修改：
- `JWT_SECRET` — 生产环境务必更换
- `LLM_SERVICE_URL` — 如果服务地址改变
- `DATABASE_URL` — 如果数据库路径改变

### 3.4 根据 GPU 配置修改 models.json

编辑 `llm_service/models.json`，调整 `device_ids` 和 `tp_size` 以匹配新机器的 GPU：

```json
[
  {
    "id": "qwen3-8b",
    "name": "Qwen3 8B (FP8)",
    "path": "models/qwen3_8b_fp8",
    "max_seq_len": 8192,
    "device": "nvidia",
    "device_ids": [0],
    "tp_size": 1
  }
]
```

- 单卡：`"device_ids": [0], "tp_size": 1`
- 双卡 Tensor Parallel：`"device_ids": [0, 1], "tp_size": 2`

---

## 第四步：构建与启动

### 4.1 构建 C++ 推理引擎

```bash
cd llm_service
xmake f --nv-gpu=y -m release
xmake build -j$(nproc)
```

### 4.2 安装 Python 依赖

```bash
cd llm_service/python
pip install -e .

cd ../api
pip install -r requirements.txt
```

### 4.3 安装前端依赖

```bash
cd web
npm install
npx prisma generate
npx prisma migrate dev
```

### 4.4 启动服务

**启动 LLM 推理服务：**

```bash
cd llm_service
python -m api.main
# 默认监听 http://localhost:8000
```

**启动前端：**

```bash
cd web
npm run dev
# 默认监听 http://localhost:3000
```

---

## 第五步：验证

```bash
bash scripts/migrate-verify.sh
```

或手动检查：

```bash
# 检查 GPU 是否可用
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 检查 C++ 引擎
python -c "import llaisys; print('llaisys 加载成功')"

# 检查 API 服务
curl http://localhost:8000/health

# 检查前端
curl -s http://localhost:3000 | head -5
```

---

## 常见问题

### Q: CUDA 版本不匹配怎么办？
xmake 构建时会自动检测本机 CUDA 版本并生成对应 GPU 代码（`add_cugencodes("native")`），所以只需重新 `xmake build` 即可。

### Q: 新机器只有一张 GPU，32B 模型能跑吗？
FP8 的 Qwen3-32B 需要约 32GB 显存，单张 4090（24GB）不够。建议用 8B 或 14B 模型，或者使用显存更大的 GPU。

### Q: 如何只迁移部分模型？
编辑 `llm_service/models.json`，只保留需要的模型，只传输对应的模型目录。

### Q: node_modules 要迁移吗？
不需要，在新机器上运行 `npm install` 即可重新安装。

### Q: dev.db 里的数据重要吗？
如果你有已创建的角色卡、世界书等数据，需要迁移 `dev.db`。否则可以用 `npm run db:reset` 重新初始化。
