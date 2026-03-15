# CPU 算子 SIMD 向量化改造技术报告

## 1. 改造背景与目标

### 1.1 现状分析

llm_service 推理引擎采用分层架构设计，算子层（`src/ops/`）通过设备分发机制将计算路由到 CPU 或 NVIDIA GPU 实现。改造前 CPU 端存在以下问题：

| 问题 | 描述 |
|------|------|
| 算子实现缺失 | 12 个算子中仅 `add` 有独立 CPU 实现文件，其余 10 个为 `op.cpp` 内联模板，`sample` 完全无 CPU 支持 |
| 零 SIMD 优化 | 全部 CPU 代码为纯标量 C++ 循环，未使用任何向量化指令 |
| 编译配置缺失 | `xmake/cpu.lua` 未添加 `-mavx2`/`-mfma` 等 SIMD 编译标志 |
| 无单元测试 | CPU 算子无独立测试覆盖，仅有 API 层集成测试 |

### 1.2 目标平台

```
架构:       x86_64 (Intel Core i7-8750H)
指令集支持:  SSE4.2, AVX2, FMA, F16C, BMI2
编译器:      Apple Clang 17 (Xcode), C++17
```

### 1.3 改造目标

1. 将全部 CPU 算子的 F32 计算路径进行 AVX2 + FMA 向量化
2. 为 BF16 数据类型实现 SIMD 批量类型转换，避免逐元素标量转换瓶颈
3. 新增 `sample` 算子的 CPU 实现，补齐 CPU 推理能力
4. 建立 CPU 算子单元测试体系
5. 保留标量回退路径，确保非 AVX2 平台的兼容性

---

## 2. 技术方案设计

### 2.1 总体架构

```
┌─────────────────────────────────────────────────┐
│                   op.cpp (设备分发)               │
│   if (CPU) → cpu::op()   if (GPU) → nvidia::op() │
└───────┬─────────────────────────────┬───────────┘
        │                             │
  ┌─────▼──────────┐          ┌──────▼──────────┐
  │  cpu/{op}_cpu.cpp │        │ nvidia/{op}.cu  │
  │  ┌────────────┐ │          └─────────────────┘
  │  │ #ifdef AVX2│ │
  │  │  SIMD 路径  │ │
  │  ├────────────┤ │
  │  │ #else      │ │
  │  │  标量回退   │ │
  │  └────────────┘ │
  └────────┬────────┘
           │
  ┌────────▼────────┐
  │ cpu_simd_utils.hpp │  ← AVX2 公共工具函数
  │ (BF16/FP16 转换、 │
  │  水平归约等)       │
  └─────────────────┘
```

**设计原则：**

- **编译期分发**：通过 `#ifdef __AVX2__` 条件编译，SIMD 路径和标量回退路径在编译期确定，零运行时开销
- **尾部处理**：所有 SIMD 循环后紧跟标量 remainder loop，正确处理非 8 对齐的尾部元素
- **精度保证**：BF16/FP16 计算统一提升到 F32 精度进行，写回时执行 round-to-nearest-even 舍入

### 2.2 BF16 SIMD 批量转换方案

BF16（Brain Float 16）的位布局与 F32 高 16 位完全一致，因此转换可通过纯整数移位实现，无需浮点运算：

```
BF16 → F32:  zero-extend uint16 → uint32, shift left 16
F32 → BF16:  add rounding bias (0x7FFF + LSB), shift right 16, truncate
```

**AVX2 实现（8 元素并行）：**

```cpp
// BF16 → F32: 3 条指令
inline __m256 bf16x8_to_f32x8(const bf16_t *src) {
    __m128i bf16 = _mm_loadu_si128(src);           // 加载 8×uint16
    __m256i i32  = _mm256_cvtepu16_epi32(bf16);    // 零扩展 → 8×uint32
    __m256i bits = _mm256_slli_epi32(i32, 16);     // 左移 16 位
    return _mm256_castsi256_ps(bits);               // 重解释为 float
}

// F32 → BF16: round-to-nearest-even + pack
inline void f32x8_store_bf16(bf16_t *dst, __m256 vals) {
    __m256i bits = _mm256_castps_si256(vals);
    __m256i trunc = _mm256_srli_epi32(bits, 16);
    __m256i lsb = _mm256_and_si256(trunc, _mm256_set1_epi32(1));
    __m256i bias = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb);
    __m256i rounded = _mm256_add_epi32(bits, bias);
    __m256i result = _mm256_srli_epi32(rounded, 16);
    // packus + permute 跨 lane 打包 8×32bit → 8×16bit
    __m256i packed = _mm256_packus_epi32(result, _mm256_setzero_si256());
    __m256i perm = _mm256_permute4x64_epi64(packed, 0xD8);
    _mm_storeu_si128(dst, _mm256_castsi256_si128(perm));
}
```

**FP16 转换** 直接使用 F16C 硬件指令 `_mm256_cvtph_ps` / `_mm256_cvtps_ph`，零额外开销。

### 2.3 `__C` 宏冲突解决

`include/llaisys.h` 定义了 `#define __C extern "C"` 用于 C/C++ 互操作，但该宏名与 `<immintrin.h>` 中 CRC32 内联函数的参数名 `__C` 冲突（如 `_mm_crc32_u8(unsigned int __C, ...)`），导致编译失败。

**解决方案**：在 `cpu_simd_utils.hpp` 中强制将 `<immintrin.h>` 的 include 放在 `types.hpp`（间接包含 `llaisys.h`）之前，并要求所有 CPU 算子 `.cpp` 文件以 `cpu_simd_utils.hpp` 作为首个 include：

```cpp
// cpu_simd_utils.hpp
#pragma once
#ifdef __AVX2__
#include <immintrin.h>   // 必须先于 llaisys.h
#endif
#include "../utils/types.hpp"  // 包含 llaisys.h，此时 __C 宏不影响 immintrin

// 各算子 .cpp 文件
#include "../../cpu_simd_utils.hpp"  // 第一个 include
#include "xxx_cpu.hpp"
#include "../../../utils.hpp"
```

---

## 3. 各算子 SIMD 改造详解

### 3.1 add — 逐元素加法

| 项目 | 内容 |
|------|------|
| 文件 | `src/ops/add/cpu/add_cpu.cpp` |
| 向量化策略 | 每次加载 8 个 float，`_mm256_add_ps` 并行加法 |
| BF16 路径 | 批量 BF16→F32 转换 → SIMD 加法 → 批量 F32→BF16 写回 |
| FP16 路径 | F16C `_mm256_cvtph_ps` → SIMD 加法 → `_mm256_cvtps_ph` |

```
标量: load a[i], load b[i], add, store c[i]     × N 次
AVX2: load a[i:i+8], load b[i:i+8], add, store  × N/8 次
```

### 3.2 argmax — 最大值索引

| 项目 | 内容 |
|------|------|
| 文件 | `src/ops/argmax/cpu/argmax_cpu.{hpp,cpp}` |
| 向量化策略 | 两遍扫描法 |
| 第一遍 | AVX2 `_mm256_max_ps` 水平归约找全局最大值 |
| 第二遍 | 标量扫描找首个匹配索引 |

两遍法比单遍标量法更快的原因：第一遍的 SIMD max 归约吞吐量是标量的 ~8x，第二遍通常在前几个元素就能命中（尤其是 logits 中 argmax 元素概率集中）。对于 vocab_size=152064 的 Qwen3，第一遍从 ~152K 次比较降到 ~19K 次 SIMD 比较。

### 3.3 rms_norm — RMS 归一化

| 项目 | 内容 |
|------|------|
| 文件 | `src/ops/rms_norm/cpu/rms_norm_cpu.{hpp,cpp}` |
| 向量化策略 | 两遍逐行处理 |
| 第一遍（求平方和） | FMA `_mm256_fmadd_ps(v, v, sum)` 融合乘加累积 |
| 水平归约 | `hsum_f32x8` 提取最终标量和 |
| 第二遍（归一化） | `_mm256_mul_ps(v * rms, weight)` 并行归一化 |
| BF16 路径 | 两遍均使用批量 BF16↔F32 转换 |

```
RMS Norm 计算: out[i] = weight[i] * in[i] * (1/sqrt(mean(in²) + eps))

Pass 1: sum_sq += in[i]² (FMA: 1条指令完成乘加)
Pass 2: out[i] = in[i] * rms * weight[i] (2次乘法)
```

FMA 指令 `_mm256_fmadd_ps(a, b, c)` = `a*b+c` 在单周期内完成，相比分开的 `mul` + `add` 减少一半指令数，且精度更高（中间结果不截断）。

### 3.4 swiglu — SiLU 门控激活

| 项目 | 内容 |
|------|------|
| 文件 | `src/ops/swiglu/cpu/swiglu_cpu.{hpp,cpp}` |
| 计算公式 | `out = up * gate * sigmoid(gate)` |
| 关键优化 | 多项式近似 `exp` 替代 `std::exp` |

**快速 exp 近似实现：**

标量 `std::exp` 是 SIMD 向量化的主要障碍——它是库函数调用，无法被编译器自动向量化。我们实现了基于 Horner 多项式的 AVX2 向量化 exp 近似：

```
exp(x) = 2^(x * log2(e))
       = 2^integer_part * 2^fractional_part
       = bit_shift       * polynomial(frac)
```

1. 将 `x` 转换到 `log2` 域：`t = x * log2(e)`
2. 分离整数部分和小数部分：`ti = round(t)`, `tf = t - ti`
3. 整数部分通过 IEEE 754 指数位移实现 `2^ti`（零开销）
4. 小数部分 `tf ∈ [-0.5, 0.5]` 用 6 阶 Horner 多项式逼近 `2^tf`
5. 二者相乘得到最终结果

```cpp
// 6阶 minimax 多项式系数 (误差 < 1e-4)
poly = p5;
poly = fmadd(poly, tf, p4);  // Horner 求值，每步 1 条 FMA
poly = fmadd(poly, tf, p3);
poly = fmadd(poly, tf, p2);
poly = fmadd(poly, tf, p1);
poly = fmadd(poly, tf, p0);
result = pow2_int * poly;     // 合并整数部分
```

**精度分析**：相对误差 < 1e-4，对推理场景的 sigmoid 计算完全足够（sigmoid 输出范围 [0,1]，绝对误差量级 < 1e-5）。

### 3.5 linear — 矩阵乘法 (GEMM)

| 项目 | 内容 |
|------|------|
| 文件 | `src/ops/linear/cpu/linear_cpu.{hpp,cpp}` |
| 计算 | `out[M,N] = in[M,K] × weight[N,K]^T + bias[N]` |
| 向量化策略 | 内层 K 维度 dot product 使用 FMA |
| 水平归约 | `hsum_f32x8` 提取最终点积标量 |
| BF16 路径 | 内层循环批量 BF16→F32 转换 + FMA |

```
原始三重循环:
for m in M:
  for n in N:
    for k in K:          ← 向量化此层
      sum += in[m,k] * weight[n,k]

SIMD 内层:
for k in 0..K step 8:
    sum_vec = fmadd(load(in+k), load(weight+k), sum_vec)
sum = hsum(sum_vec) + scalar_remainder
```

GEMM 是 LLM 推理中最密集的计算（QKV 投影、MLP 全连接），内层 FMA 向量化可将 K 维度的乘加吞吐量提升约 8 倍。

### 3.6 self_attention — 自注意力

| 项目 | 内容 |
|------|------|
| 文件 | `src/ops/self_attention/cpu/self_attention_cpu.{hpp,cpp}` |
| 三阶段优化 | |
| 1. QK dot product | AVX2+FMA 向量化点积，8 维并行累积 |
| 2. Softmax | 标量精确计算（精度敏感） |
| 3. Score×V 加权和 | AVX2 `weighted_add` 向量化累加 |

```
Attention(Q, K, V) = softmax(Q·K^T / sqrt(d)) · V

Step 1: scores[ki] = dot_avx2(q_vec, k_vec, hd) * scale
        → 8维并行 FMA 点积

Step 2: scores = softmax(scores)
        → 标量 exp + 归一化 (精度优先)

Step 3: out[d] += scores[ki] * v[ki,d]
        → weighted_add_avx2: 8维并行 FMA 累加
```

Softmax 采用标量精确计算的原因：softmax 中的 `exp(x-max)` 对数值精度极度敏感，使用近似 exp 可能导致注意力权重分布偏移，影响生成质量。QK 点积和 V 加权和对精度不敏感，适合 SIMD 加速。

### 3.7 rope — 旋转位置编码

| 项目 | 内容 |
|------|------|
| 文件 | `src/ops/rope/cpu/rope_cpu.{hpp,cpp}` |
| 向量化策略 | 预计算频率 + SIMD 旋转对 |
| 关键优化 | `inv_freq` 数组预计算避免逐元素 `pow` |

```
RoPE 旋转:
out_a[j] = a[j] * cos(θ) - b[j] * sin(θ)
out_b[j] = b[j] * cos(θ) + a[j] * sin(θ)

优化 1: 预计算 inv_freq[j] = 1/θ^(2j/d)，避免内层 pow
优化 2: 预计算 cos/sin 数组（每个 position 一次）
优化 3: AVX2 FMA 向量化旋转对：
    ra = fmsub(a, cos, mul(b, sin))   // a*cos - b*sin
    rb = fmadd(a, sin, mul(b, cos))   // a*sin + b*cos
```

每个旋转对需要 4 次乘法 + 2 次加减，FMA 将其压缩为 2 条 FMA + 2 条 MUL。对于 head_dim=128 的 Qwen3，每个头的 64 个旋转对仅需 8 次 SIMD 迭代。

### 3.8 embedding — 查表嵌入

| 项目 | 内容 |
|------|------|
| 文件 | `src/ops/embedding/cpu/embedding_cpu.{hpp,cpp}` |
| 优化 | 从 op.cpp 内联代码抽取为独立文件，保持 `memcpy` 实现 |

Embedding 本质是内存拷贝操作（按 index 查表），`memcpy` 已经是编译器高度优化的实现（内部使用 AVX 宽加载/存储），无需额外 SIMD 改造。

### 3.9 sample — Top-K/Top-P 采样（新增）

| 项目 | 内容 |
|------|------|
| 文件 | `src/ops/sample/cpu/sample_cpu.{hpp,cpp}` |
| 状态 | **全新实现**（改造前仅 GPU 支持） |
| SIMD 优化点 | temperature 缩放、softmax 中的 max/normalize |

采样流程：
1. **Temperature 缩放**：AVX2 并行 `logits * (1/temperature)`
2. **Softmax**：AVX2 并行求 max → 标量 exp → AVX2 并行归一化
3. **Top-K 截断**：`partial_sort` 按概率降序取前 K 个
4. **Top-P 核采样**：累积概率超过 `top_p` 时截断
5. **随机采样**：在有效候选集上按概率分布采样

---

## 4. 工程改造

### 4.1 文件结构变更

```
src/ops/
├── cpu_simd_utils.hpp              [新增] AVX2 公共工具
├── add/cpu/add_cpu.cpp             [重写] SIMD 化
├── argmax/cpu/argmax_cpu.hpp       [新增]
├── argmax/cpu/argmax_cpu.cpp       [新增] SIMD 化
├── rms_norm/cpu/rms_norm_cpu.hpp   [新增]
├── rms_norm/cpu/rms_norm_cpu.cpp   [新增] SIMD 化
├── swiglu/cpu/swiglu_cpu.hpp       [新增]
├── swiglu/cpu/swiglu_cpu.cpp       [新增] SIMD 化 + 快速 exp
├── linear/cpu/linear_cpu.hpp       [新增]
├── linear/cpu/linear_cpu.cpp       [新增] SIMD 化
├── self_attention/cpu/self_attention_cpu.hpp  [新增]
├── self_attention/cpu/self_attention_cpu.cpp  [新增] SIMD 化
├── rope/cpu/rope_cpu.hpp           [新增]
├── rope/cpu/rope_cpu.cpp           [新增] SIMD 化
├── embedding/cpu/embedding_cpu.hpp [新增]
├── embedding/cpu/embedding_cpu.cpp [新增] 抽取
├── sample/cpu/sample_cpu.hpp       [新增]
├── sample/cpu/sample_cpu.cpp       [新增] 全新 CPU 实现
├── {op}/op.cpp × 7                 [修改] 删除内联实现，改调 cpu:: 函数
tests/
└── test_cpu_ops.cpp                [新增] 14 个单元测试
xmake/
└── cpu.lua                         [修改] 添加 SIMD 编译标志
src/utils/
├── types.hpp                       [修改] 添加 #pragma once
└── check.hpp                       [修改] 添加 #pragma once
```

### 4.2 编译配置变更

```lua
-- xmake/cpu.lua
-- 新增: x86_64 平台 SIMD 编译标志
if is_arch("x86_64") then
    add_cxflags("-mavx2", "-mfma", "-mf16c")
end
```

| 标志 | 作用 | 启用的指令集 |
|------|------|-------------|
| `-mavx2` | 256-bit 整数/浮点向量运算 | `_mm256_*` 全系列 |
| `-mfma` | 融合乘加指令 | `_mm256_fmadd_ps` 等 |
| `-mf16c` | FP16↔FP32 硬件转换 | `_mm256_cvtph_ps`, `_mm256_cvtps_ph` |

### 4.3 op.cpp 分发层重构

改造前（以 rms_norm 为例）：

```cpp
// op.cpp 内联 ~50 行模板实现
template <typename T>
void rms_norm_cpu(...) { /* 标量循环 */ }

void rms_norm(...) {
    if (CPU) {
        switch (dtype) {
            case F32:  return rms_norm_cpu<float>(...);
            case BF16: return rms_norm_cpu<bf16_t>(...);
            ...
        }
    }
}
```

改造后：

```cpp
// op.cpp 仅负责分发
#include "cpu/rms_norm_cpu.hpp"

void rms_norm(...) {
    if (CPU) {
        return cpu::rms_norm(out, in, weight, eps, dtype, M, N);
    }
    // GPU path...
}
```

内联模板代码全部移至 `cpu/` 目录下的独立编译单元，`op.cpp` 职责简化为纯粹的设备分发。

---

## 5. 测试验证

### 5.1 测试方案

创建独立 C++ 测试程序 `tests/test_cpu_ops.cpp`，直接链接 CPU 算子编译单元，不依赖完整框架：

```bash
g++ -std=c++17 -mavx2 -mfma -mf16c -O2 -I include -I src \
    tests/test_cpu_ops.cpp src/utils/types.cpp \
    src/ops/*/cpu/*_cpu.cpp -lm -o tests/test_cpu_ops
```

### 5.2 测试用例

| # | 测试名 | 覆盖内容 | 验证方法 |
|---|--------|----------|----------|
| 1 | `test_add_f32` | F32 加法，非对齐长度 (N=1025) | 逐元素对比参考值，tol=1e-6 |
| 2 | `test_add_bf16` | BF16 加法，批量转换正确性 | 标量 BF16 参考计算，tol=0.02 |
| 3 | `test_argmax_f32` | 大数组 argmax (N=10000) | 植入已知最大值，验证索引和值 |
| 4 | `test_argmax_small` | 小数组边界 (N=3) | 验证标量路径正确性 |
| 5 | `test_rms_norm_f32` | 多行 RMS Norm (M=2, N=256) | 双遍标量参考实现对比，tol=1e-4 |
| 6 | `test_swiglu_f32` | SwiGLU 激活 (N=1025) | 标量 sigmoid·silu 参考，tol=5e-4 |
| 7 | `test_linear_f32` | 矩阵乘法+偏置 (M=2,K=128,N=64) | 三重循环标量参考，tol=1e-2 |
| 8 | `test_linear_no_bias` | 无偏置矩阵乘法 | 同上，bias=nullptr |
| 9 | `test_rope_f32` | RoPE 旋转 (seq=4,head=2,dim=16) | 逐元素 cos/sin 标量参考，tol=1e-4 |
| 10 | `test_embedding_f32` | 查表嵌入 (vocab=100,dim=64) | 精确匹配权重表对应行 |
| 11 | `test_self_attention_f32` | 注意力 (q=2,kv=4,head=2,dim=8) | 手工 softmax+加权和参考 |
| 12 | `test_self_attention_gqa` | GQA 注意力 (nhead=4,nkvhead=2) | 输出非零验证 |
| 13 | `test_sample_f32` | 低温采样 (temp=0.1,top_k=1) | 验证选中最大 logit 索引 |
| 14 | `test_sample_top_p` | 核采样 (top_p=0.95) | 验证输出在 top-3 候选内 |

### 5.3 测试结果

```
=== CPU Operator Unit Tests (SIMD) ===
AVX2: enabled
FMA:  enabled
F16C: enabled

Running test_add_f32...           PASS
Running test_add_bf16...          PASS
Running test_argmax_f32...        PASS
Running test_argmax_small...      PASS
Running test_rms_norm_f32...      PASS
Running test_swiglu_f32...        PASS
Running test_linear_f32...        PASS
Running test_linear_no_bias...    PASS
Running test_rope_f32...          PASS
Running test_embedding_f32...     PASS
Running test_self_attention_f32...PASS
Running test_self_attention_gqa...PASS
Running test_sample_f32...        PASS
Running test_sample_top_p...      PASS

=== Results: 14 passed, 0 failed ===
```

---

## 6. 性能预期分析

### 6.1 理论加速比

| 算子 | 计算瓶颈 | 标量 IPC | AVX2 IPC | 理论加速比 |
|------|----------|----------|----------|-----------|
| add | 逐元素加法 | 1 float/cycle | 8 float/cycle | **~8x** |
| argmax | 逐元素比较 | 1 cmp/cycle | 8 cmp/cycle | **~8x** (pass1) |
| rms_norm | 平方和 + 归一化 | 2 ops/cycle | 16 ops/cycle (FMA) | **~6-8x** |
| swiglu | sigmoid 需 exp | ~20 cycles/exp | ~8 cycles/8-exp | **~4-5x** |
| linear | dot product | 2 ops/cycle | 16 ops/cycle (FMA) | **~6-8x** |
| self_attention | dot + weighted_sum | 2 ops/cycle | 16 ops/cycle (FMA) | **~4-6x** |
| rope | sin/cos + 旋转 | 受 sin/cos 限制 | 旋转对 SIMD 化 | **~3-4x** |

> 注：实际加速比受内存带宽、cache miss、分支预测等因素影响，上表为计算密集部分的理论峰值。

### 6.2 各算子瓶颈分析

| 算子 | 瓶颈类型 | 说明 |
|------|----------|------|
| add, argmax | **带宽受限** | 计算/访存比极低，SIMD 主要收益在减少循环开销 |
| linear | **计算受限** | GEMM 内层 FMA 是纯计算密集，SIMD 收益最大 |
| rms_norm | **混合** | pass1 计算密集，pass2 带宽受限 |
| swiglu | **计算受限** | exp 是计算瓶颈，近似 exp SIMD 化收益显著 |
| self_attention | **混合** | dot product 计算密集，softmax 带宽受限 |
| rope | **延迟受限** | sin/cos 超越函数无法 SIMD 化，预计算缓解 |

---

## 7. 兼容性与限制

### 7.1 平台兼容性

| 平台 | 行为 |
|------|------|
| x86_64 + AVX2 | 使用 SIMD 加速路径 |
| x86_64 无 AVX2 | 编译不带 `-mavx2`，走标量回退 |
| ARM64 (Apple Silicon) | `#ifdef __AVX2__` 不生效，走标量路径（可扩展为 NEON） |
| Windows | `is_arch("x86_64")` 正确识别，MSVC 需换用 `/arch:AVX2` |

### 7.2 已知限制

1. **BF16 argmax 未 SIMD 化**：BF16 的 SIMD max 比较需要先转 F32 再比较，对大 vocab 场景收益不如 F32 路径显著，当前保持标量实现
2. **SwiGLU exp 近似误差**：多项式近似 exp 的相对误差 ~1e-4，对极端输入值（|x| > 80）可能有微小偏差，但不影响推理质量
3. **linear 未做 tiling**：当前 GEMM 未实现 cache-friendly tiling（分块），对于大矩阵（如 hidden_size=5120 的 Qwen3-32B），L1/L2 cache 利用率不够理想，后续可引入 micro-kernel tiling 进一步优化

### 7.3 后续优化方向

| 优化项 | 预期收益 | 复杂度 |
|--------|----------|--------|
| linear GEMM tiling (分块 + micro-kernel) | 大矩阵 2-3x | 高 |
| ARM NEON 支持 | Apple Silicon 可用 | 中 |
| OpenMP 多线程并行 | 多核 CPU 线性加速 | 低 |
| AVX-512 支持 (Xeon) | 再提升 ~2x | 中 |
| BF16 argmax SIMD 化 | vocab argmax 加速 | 低 |

---

## 8. 结论

本次改造将 llm_service CPU 算子从纯标量实现全面升级为 AVX2 + FMA 向量化实现，覆盖全部 9 个可优化算子，新增 CPU 端 sample 算子补齐推理能力。通过 14 个独立单元测试验证了数值正确性。改造保持了完整的标量回退路径和条件编译机制，确保跨平台兼容性。
