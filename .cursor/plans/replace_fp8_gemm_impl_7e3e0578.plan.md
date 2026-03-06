---
name: Replace FP8 GEMM impl
overview: 将 linear_fp8 从分块循环 (N/128 次 dequant+GEMM) 改为一次性全量反量化 + 单次 cuBLAS GEMM，消除 decode 阶段的 kernel launch 瓶颈。
todos: []
isProject: false
---

# 替换 FP8 GEMM 实现：