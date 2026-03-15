// Standalone CPU operator unit test
// Compile: g++ -std=c++17 -mavx2 -mfma -mf16c -O2 -I../include -I../src tests/test_cpu_ops.cpp
//          src/utils/types.cpp src/ops/add/cpu/add_cpu.cpp src/ops/argmax/cpu/argmax_cpu.cpp
//          src/ops/rms_norm/cpu/rms_norm_cpu.cpp src/ops/swiglu/cpu/swiglu_cpu.cpp
//          src/ops/linear/cpu/linear_cpu.cpp src/ops/rope/cpu/rope_cpu.cpp
//          src/ops/embedding/cpu/embedding_cpu.cpp src/ops/sample/cpu/sample_cpu.cpp
//          src/ops/self_attention/cpu/self_attention_cpu.cpp -lm -o test_cpu_ops

// Include immintrin.h BEFORE llaisys.h to avoid __C macro conflict
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "../src/utils/types.hpp"
#include "../src/utils/check.hpp"

// CPU op headers
#include "../src/ops/add/cpu/add_cpu.hpp"
#include "../src/ops/argmax/cpu/argmax_cpu.hpp"
#include "../src/ops/rms_norm/cpu/rms_norm_cpu.hpp"
#include "../src/ops/swiglu/cpu/swiglu_cpu.hpp"
#include "../src/ops/linear/cpu/linear_cpu.hpp"
#include "../src/ops/rope/cpu/rope_cpu.hpp"
#include "../src/ops/embedding/cpu/embedding_cpu.hpp"
#include "../src/ops/sample/cpu/sample_cpu.hpp"
#include "../src/ops/self_attention/cpu/self_attention_cpu.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace llaisys;
using namespace llaisys::ops::cpu;

// ============================================================
// Test utilities
// ============================================================
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                           \
    do {                                                                 \
        if (!(cond)) {                                                   \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__);            \
            return false;                                                \
        }                                                                \
    } while (0)

#define RUN_TEST(fn)                                                     \
    do {                                                                 \
        printf("Running %s...\n", #fn);                                  \
        if (fn()) {                                                      \
            printf("  PASS\n");                                          \
            g_tests_passed++;                                            \
        } else {                                                         \
            g_tests_failed++;                                            \
        }                                                                \
    } while (0)

static float randf(std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    return dist(rng);
}

static bool approx_eq(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol + tol * std::fabs(b);
}

// ============================================================
// Test: add
// ============================================================
static bool test_add_f32() {
    const size_t N = 1025; // non-aligned size to test remainder loop
    std::vector<float> a(N), b(N), c(N), ref(N);
    std::mt19937 rng(42);

    for (size_t i = 0; i < N; i++) {
        a[i] = randf(rng);
        b[i] = randf(rng);
        ref[i] = a[i] + b[i];
    }

    add(reinterpret_cast<std::byte *>(c.data()),
        reinterpret_cast<const std::byte *>(a.data()),
        reinterpret_cast<const std::byte *>(b.data()),
        LLAISYS_DTYPE_F32, N);

    for (size_t i = 0; i < N; i++) {
        TEST_ASSERT(approx_eq(c[i], ref[i], 1e-6f), "add f32 mismatch");
    }
    return true;
}

static bool test_add_bf16() {
    const size_t N = 1025;
    std::vector<bf16_t> a(N), b(N), c(N);
    std::mt19937 rng(42);

    for (size_t i = 0; i < N; i++) {
        a[i] = utils::cast<bf16_t>(randf(rng));
        b[i] = utils::cast<bf16_t>(randf(rng));
    }

    add(reinterpret_cast<std::byte *>(c.data()),
        reinterpret_cast<const std::byte *>(a.data()),
        reinterpret_cast<const std::byte *>(b.data()),
        LLAISYS_DTYPE_BF16, N);

    for (size_t i = 0; i < N; i++) {
        float expected = utils::cast<float>(a[i]) + utils::cast<float>(b[i]);
        float actual = utils::cast<float>(c[i]);
        TEST_ASSERT(approx_eq(actual, expected, 0.02f), "add bf16 mismatch");
    }
    return true;
}

// ============================================================
// Test: argmax
// ============================================================
static bool test_argmax_f32() {
    const size_t N = 10000;
    std::vector<float> vals(N);
    std::mt19937 rng(42);

    for (size_t i = 0; i < N; i++) {
        vals[i] = randf(rng);
    }
    // Plant known max
    vals[7777] = 100.0f;

    int64_t max_idx = -1;
    float max_val = 0.0f;

    argmax(reinterpret_cast<std::byte *>(&max_idx),
           reinterpret_cast<std::byte *>(&max_val),
           reinterpret_cast<const std::byte *>(vals.data()),
           LLAISYS_DTYPE_F32, N);

    TEST_ASSERT(max_idx == 7777, "argmax f32 wrong index");
    TEST_ASSERT(approx_eq(max_val, 100.0f), "argmax f32 wrong value");
    return true;
}

static bool test_argmax_small() {
    // Edge case: very small array
    float vals[] = {-1.0f, 3.0f, 2.0f};
    int64_t max_idx = -1;
    float max_val = 0.0f;

    argmax(reinterpret_cast<std::byte *>(&max_idx),
           reinterpret_cast<std::byte *>(&max_val),
           reinterpret_cast<const std::byte *>(vals),
           LLAISYS_DTYPE_F32, 3);

    TEST_ASSERT(max_idx == 1, "argmax small wrong index");
    TEST_ASSERT(approx_eq(max_val, 3.0f), "argmax small wrong value");
    return true;
}

// ============================================================
// Test: rms_norm
// ============================================================
static bool test_rms_norm_f32() {
    const size_t M = 2, N = 256;
    std::vector<float> in(M * N), out(M * N), weight(N);
    std::mt19937 rng(42);

    for (size_t i = 0; i < M * N; i++) in[i] = randf(rng);
    for (size_t i = 0; i < N; i++) weight[i] = randf(rng);

    float eps = 1e-6f;
    rms_norm(reinterpret_cast<std::byte *>(out.data()),
             reinterpret_cast<const std::byte *>(in.data()),
             reinterpret_cast<const std::byte *>(weight.data()),
             eps, LLAISYS_DTYPE_F32, M, N);

    // Verify against reference
    for (size_t m = 0; m < M; m++) {
        float sum_sq = 0.0f;
        for (size_t n = 0; n < N; n++) {
            sum_sq += in[m * N + n] * in[m * N + n];
        }
        float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(N) + eps);

        for (size_t n = 0; n < N; n++) {
            float expected = weight[n] * in[m * N + n] * rms;
            TEST_ASSERT(approx_eq(out[m * N + n], expected, 1e-4f), "rms_norm f32 mismatch");
        }
    }
    return true;
}

// ============================================================
// Test: swiglu
// ============================================================
static bool test_swiglu_f32() {
    const size_t N = 1025;
    std::vector<float> gate(N), up(N), out(N);
    std::mt19937 rng(42);

    for (size_t i = 0; i < N; i++) {
        gate[i] = randf(rng);
        up[i] = randf(rng);
    }

    swiglu(reinterpret_cast<std::byte *>(out.data()),
           reinterpret_cast<const std::byte *>(gate.data()),
           reinterpret_cast<const std::byte *>(up.data()),
           LLAISYS_DTYPE_F32, N);

    for (size_t i = 0; i < N; i++) {
        float g = gate[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-g));
        float expected = up[i] * g * sigmoid;
        // Relaxed tolerance due to approximate exp in SIMD path
        TEST_ASSERT(approx_eq(out[i], expected, 5e-4f), "swiglu f32 mismatch");
    }
    return true;
}

// ============================================================
// Test: linear
// ============================================================
static bool test_linear_f32() {
    const size_t M = 2, K = 128, N = 64;
    std::vector<float> in(M * K), weight(N * K), bias(N), out(M * N);
    std::mt19937 rng(42);

    for (auto &v : in) v = randf(rng);
    for (auto &v : weight) v = randf(rng);
    for (auto &v : bias) v = randf(rng);

    linear(reinterpret_cast<std::byte *>(out.data()),
           reinterpret_cast<const std::byte *>(in.data()),
           reinterpret_cast<const std::byte *>(weight.data()),
           reinterpret_cast<const std::byte *>(bias.data()),
           LLAISYS_DTYPE_F32, M, K, N);

    // Verify: out[m,n] = sum_k(in[m,k] * weight[n,k]) + bias[n]
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float expected = 0.0f;
            for (size_t k = 0; k < K; k++) {
                expected += in[m * K + k] * weight[n * K + k];
            }
            expected += bias[n];
            // FMA may give slightly different results
            TEST_ASSERT(approx_eq(out[m * N + n], expected, 1e-2f), "linear f32 mismatch");
        }
    }
    return true;
}

static bool test_linear_no_bias() {
    const size_t M = 1, K = 64, N = 32;
    std::vector<float> in(M * K), weight(N * K), out(M * N);
    std::mt19937 rng(123);

    for (auto &v : in) v = randf(rng);
    for (auto &v : weight) v = randf(rng);

    linear(reinterpret_cast<std::byte *>(out.data()),
           reinterpret_cast<const std::byte *>(in.data()),
           reinterpret_cast<const std::byte *>(weight.data()),
           nullptr, LLAISYS_DTYPE_F32, M, K, N);

    for (size_t n = 0; n < N; n++) {
        float expected = 0.0f;
        for (size_t k = 0; k < K; k++) {
            expected += in[k] * weight[n * K + k];
        }
        TEST_ASSERT(approx_eq(out[n], expected, 1e-2f), "linear no-bias mismatch");
    }
    return true;
}

// ============================================================
// Test: rope
// ============================================================
static bool test_rope_f32() {
    const size_t seq_len = 4, n_head = 2, head_dim = 16;
    const size_t total = seq_len * n_head * head_dim;
    std::vector<float> in(total), out(total);
    std::vector<int64_t> pos_ids = {0, 1, 2, 3};
    float theta = 10000.0f;
    std::mt19937 rng(42);

    for (auto &v : in) v = randf(rng);

    rope(reinterpret_cast<std::byte *>(out.data()),
         reinterpret_cast<const std::byte *>(in.data()),
         reinterpret_cast<const std::byte *>(pos_ids.data()),
         theta, LLAISYS_DTYPE_F32, seq_len, n_head, head_dim);

    // Verify against reference
    size_t half_dim = head_dim / 2;
    for (size_t s = 0; s < seq_len; s++) {
        float pos = static_cast<float>(pos_ids[s]);
        for (size_t h = 0; h < n_head; h++) {
            for (size_t j = 0; j < half_dim; j++) {
                float freq = pos / std::pow(theta, 2.0f * j / static_cast<float>(head_dim));
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                size_t idx_a = s * n_head * head_dim + h * head_dim + j;
                size_t idx_b = idx_a + half_dim;

                float expected_a = in[idx_a] * cos_val - in[idx_b] * sin_val;
                float expected_b = in[idx_b] * cos_val + in[idx_a] * sin_val;

                TEST_ASSERT(approx_eq(out[idx_a], expected_a, 1e-4f), "rope f32 a mismatch");
                TEST_ASSERT(approx_eq(out[idx_b], expected_b, 1e-4f), "rope f32 b mismatch");
            }
        }
    }
    return true;
}

// ============================================================
// Test: embedding
// ============================================================
static bool test_embedding_f32() {
    const size_t vocab_size = 100, hidden_size = 64, seq_len = 5;
    std::vector<float> weight(vocab_size * hidden_size);
    std::vector<float> out(seq_len * hidden_size);
    std::vector<int64_t> indices = {0, 42, 99, 7, 50};
    std::mt19937 rng(42);

    for (auto &v : weight) v = randf(rng);

    embedding(reinterpret_cast<std::byte *>(out.data()),
              reinterpret_cast<const std::byte *>(indices.data()),
              reinterpret_cast<const std::byte *>(weight.data()),
              LLAISYS_DTYPE_F32, seq_len, hidden_size);

    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < hidden_size; j++) {
            float expected = weight[indices[i] * hidden_size + j];
            TEST_ASSERT(out[i * hidden_size + j] == expected, "embedding mismatch");
        }
    }
    return true;
}

// ============================================================
// Test: self_attention
// ============================================================
static bool test_self_attention_f32() {
    // Small attention: qlen=2, kvlen=4, nhead=2, nkvhead=2, hd=8
    const size_t qlen = 2, kvlen = 4, nhead = 2, nkvhead = 2, hd = 8;
    std::vector<float> q(qlen * nhead * hd), k(kvlen * nkvhead * hd),
        v(kvlen * nkvhead * hd), attn(qlen * nhead * hd);
    std::mt19937 rng(42);

    for (auto &val : q) val = randf(rng);
    for (auto &val : k) val = randf(rng);
    for (auto &val : v) val = randf(rng);

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    self_attention(reinterpret_cast<std::byte *>(attn.data()),
                   reinterpret_cast<const std::byte *>(q.data()),
                   reinterpret_cast<const std::byte *>(k.data()),
                   reinterpret_cast<const std::byte *>(v.data()),
                   scale, LLAISYS_DTYPE_F32, qlen, kvlen, nhead, nkvhead, hd);

    // Reference: manually compute for head 0, query 0
    {
        size_t h = 0, qi = 0;
        size_t causal_limit = qi + (kvlen - qlen); // 0 + 2 = 2

        std::vector<float> scores(kvlen);
        float max_score = -std::numeric_limits<float>::infinity();

        for (size_t ki = 0; ki < kvlen; ki++) {
            if (ki > causal_limit) {
                scores[ki] = -std::numeric_limits<float>::infinity();
            } else {
                float dot = 0.0f;
                for (size_t d = 0; d < hd; d++) {
                    dot += q[qi * nhead * hd + h * hd + d] * k[ki * nkvhead * hd + h * hd + d];
                }
                scores[ki] = dot * scale;
            }
            if (scores[ki] > max_score) max_score = scores[ki];
        }

        float sum_exp = 0.0f;
        for (size_t ki = 0; ki < kvlen; ki++) {
            if (scores[ki] > -1e30f) {
                scores[ki] = std::exp(scores[ki] - max_score);
                sum_exp += scores[ki];
            } else {
                scores[ki] = 0.0f;
            }
        }
        for (auto &s : scores) s /= sum_exp;

        for (size_t d = 0; d < hd; d++) {
            float expected = 0.0f;
            for (size_t ki = 0; ki < kvlen; ki++) {
                if (scores[ki] > 0.0f) {
                    expected += scores[ki] * v[ki * nkvhead * hd + h * hd + d];
                }
            }
            float actual = attn[qi * nhead * hd + h * hd + d];
            TEST_ASSERT(approx_eq(actual, expected, 1e-4f), "self_attention f32 mismatch");
        }
    }

    // Basic sanity: output should not be all zeros
    float sum = 0.0f;
    for (auto val : attn) sum += std::fabs(val);
    TEST_ASSERT(sum > 0.0f, "self_attention output all zeros");

    return true;
}

// ============================================================
// Test: self_attention with GQA (grouped-query attention)
// ============================================================
static bool test_self_attention_gqa() {
    // nhead=4, nkvhead=2 -> 2 query heads per KV head
    const size_t qlen = 1, kvlen = 3, nhead = 4, nkvhead = 2, hd = 8;
    std::vector<float> q(qlen * nhead * hd), k(kvlen * nkvhead * hd),
        v(kvlen * nkvhead * hd), attn(qlen * nhead * hd);
    std::mt19937 rng(123);

    for (auto &val : q) val = randf(rng);
    for (auto &val : k) val = randf(rng);
    for (auto &val : v) val = randf(rng);

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    self_attention(reinterpret_cast<std::byte *>(attn.data()),
                   reinterpret_cast<const std::byte *>(q.data()),
                   reinterpret_cast<const std::byte *>(k.data()),
                   reinterpret_cast<const std::byte *>(v.data()),
                   scale, LLAISYS_DTYPE_F32, qlen, kvlen, nhead, nkvhead, hd);

    float sum = 0.0f;
    for (auto val : attn) sum += std::fabs(val);
    TEST_ASSERT(sum > 0.0f, "self_attention GQA output all zeros");
    return true;
}

// ============================================================
// Test: sample
// ============================================================
static bool test_sample_f32() {
    const size_t V = 100;
    std::vector<float> logits(V, 0.0f);
    std::vector<float> workspace(V);
    int64_t output_idx = -1;

    // Make one logit much larger
    logits[42] = 10.0f;

    sample(reinterpret_cast<std::byte *>(&output_idx),
           reinterpret_cast<const std::byte *>(logits.data()),
           reinterpret_cast<std::byte *>(workspace.data()),
           LLAISYS_DTYPE_F32, V,
           0.1f, 1, 1.0f, 12345);

    // With temp=0.1 and top_k=1, should always pick the max
    TEST_ASSERT(output_idx == 42, "sample f32 should pick max logit with low temp");
    return true;
}

static bool test_sample_top_p() {
    const size_t V = 10;
    std::vector<float> logits(V);
    std::vector<float> workspace(V);
    int64_t output_idx = -1;

    // Set up logits so after softmax, top 3 have ~98% probability
    logits[0] = 5.0f;
    logits[1] = 4.5f;
    logits[2] = 4.0f;
    for (size_t i = 3; i < V; i++) logits[i] = -10.0f;

    sample(reinterpret_cast<std::byte *>(&output_idx),
           reinterpret_cast<const std::byte *>(logits.data()),
           reinterpret_cast<std::byte *>(workspace.data()),
           LLAISYS_DTYPE_F32, V,
           1.0f, 0, 0.95f, 42);

    // Should pick from top 3
    TEST_ASSERT(output_idx >= 0 && output_idx <= 2, "sample top_p should pick from top tokens");
    return true;
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("=== CPU Operator Unit Tests (SIMD) ===\n");
#ifdef __AVX2__
    printf("AVX2: enabled\n");
#else
    printf("AVX2: disabled (scalar fallback)\n");
#endif
#ifdef __FMA__
    printf("FMA:  enabled\n");
#else
    printf("FMA:  disabled\n");
#endif
#ifdef __F16C__
    printf("F16C: enabled\n");
#else
    printf("F16C: disabled\n");
#endif
    printf("\n");

    RUN_TEST(test_add_f32);
    RUN_TEST(test_add_bf16);
    RUN_TEST(test_argmax_f32);
    RUN_TEST(test_argmax_small);
    RUN_TEST(test_rms_norm_f32);
    RUN_TEST(test_swiglu_f32);
    RUN_TEST(test_linear_f32);
    RUN_TEST(test_linear_no_bias);
    RUN_TEST(test_rope_f32);
    RUN_TEST(test_embedding_f32);
    RUN_TEST(test_self_attention_f32);
    RUN_TEST(test_self_attention_gqa);
    RUN_TEST(test_sample_f32);
    RUN_TEST(test_sample_top_p);

    printf("\n=== Results: %d passed, %d failed ===\n", g_tests_passed, g_tests_failed);
    return g_tests_failed > 0 ? 1 : 0;
}
