#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#ifdef ENABLE_NVIDIA_API
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#endif

namespace llaisys {

// ============================================================================
// NVTX range helpers (for Nsight Systems timeline visualization)
// ============================================================================

#ifdef ENABLE_NVIDIA_API

struct NvtxRange {
    explicit NvtxRange(const char *name) { nvtxRangePush(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
};

#define LLAISYS_NVTX_RANGE(name)  llaisys::NvtxRange _nvtx_rng_(name)
#define LLAISYS_NVTX_RANGE_FN()   llaisys::NvtxRange _nvtx_fn_(__func__)

#else

#define LLAISYS_NVTX_RANGE(name)  ((void)0)
#define LLAISYS_NVTX_RANGE_FN()   ((void)0)

#endif

// ============================================================================
// GPU-safe timer: uses cudaDeviceSynchronize + host clock to avoid
// cross-device CUDA event issues in TP scenarios.
// ============================================================================

#ifdef ENABLE_NVIDIA_API

class CudaTimer {
public:
    void start() {
        cudaDeviceSynchronize();
        start_ = std::chrono::steady_clock::now();
    }
    void stop() {
        cudaDeviceSynchronize();
        stop_ = std::chrono::steady_clock::now();
    }
    float elapsed_ms() {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_).count();
        return us / 1000.0f;
    }
private:
    std::chrono::steady_clock::time_point start_, stop_;
};

#else

class CudaTimer {
public:
    void start() {}
    void stop()  {}
    float elapsed_ms() { return 0.0f; }
};

#endif

// ============================================================================
// InferProfiler: collects per-op timing and prints summary table
// ============================================================================

class InferProfiler {
public:
    void setEnabled(bool e) { enabled_ = e; }
    bool enabled() const { return enabled_; }

    void beginInfer(size_t num_tokens, bool is_prefill) {
        if (!enabled_) return;
        records_.clear();
        num_tokens_ = num_tokens;
        is_prefill_ = is_prefill;
        total_timer_.start();
    }

    void record(const char *op_name, size_t layer_idx, float ms) {
        if (!enabled_) return;
        records_.push_back({op_name, layer_idx, ms});
    }

    void endInfer() {
        if (!enabled_) return;
        total_timer_.stop();
        float total_ms = total_timer_.elapsed_ms();

        const char *phase = is_prefill_ ? "prefill" : "decode";
        std::fprintf(stderr,
            "\n[Profiler] %s  tokens=%zu  total=%.3f ms\n",
            phase, num_tokens_, total_ms);

        struct OpSummary { const char *name; float total_ms; float max_ms; size_t count; };
        std::vector<OpSummary> summary;

        for (auto &r : records_) {
            OpSummary *found = nullptr;
            for (auto &s : summary) {
                if (std::strcmp(s.name, r.op_name) == 0) { found = &s; break; }
            }
            if (found) {
                found->total_ms += r.ms;
                if (r.ms > found->max_ms) found->max_ms = r.ms;
                found->count++;
            } else {
                summary.push_back({r.op_name, r.ms, r.ms, 1});
            }
        }

        std::sort(summary.begin(), summary.end(),
                  [](const OpSummary &a, const OpSummary &b) { return a.total_ms > b.total_ms; });

        std::fprintf(stderr, "  %-24s %10s %10s %10s %6s %7s\n",
                     "Op", "Total(ms)", "Avg(ms)", "Max(ms)", "Count", "Pct");
        std::fprintf(stderr, "  %-24s %10s %10s %10s %6s %7s\n",
                     "------------------------", "----------", "----------", "----------", "------", "-------");
        for (auto &s : summary) {
            float avg = s.total_ms / (float)s.count;
            float pct = (total_ms > 0) ? (s.total_ms / total_ms * 100.0f) : 0.0f;
            std::fprintf(stderr, "  %-24s %10.3f %10.3f %10.3f %6zu %6.1f%%\n",
                         s.name, s.total_ms, avg, s.max_ms, s.count, pct);
        }
        std::fprintf(stderr, "\n");
    }

private:
    struct Record { const char *op_name; size_t layer_idx; float ms; };
    bool enabled_ = false;
    size_t num_tokens_ = 0;
    bool is_prefill_ = false;
    CudaTimer total_timer_;
    std::vector<Record> records_;
};

// ============================================================================
// ScopedOpTimer: RAII helper that times a scope and records to profiler
// ============================================================================

class ScopedOpTimer {
public:
    ScopedOpTimer(InferProfiler &prof, const char *name, size_t layer)
        : prof_(prof), name_(name), layer_(layer), active_(prof.enabled()) {
        if (active_) timer_.start();
    }
    ~ScopedOpTimer() {
        if (active_) {
            timer_.stop();
            prof_.record(name_, layer_, timer_.elapsed_ms());
        }
    }
    ScopedOpTimer(const ScopedOpTimer &) = delete;
    ScopedOpTimer &operator=(const ScopedOpTimer &) = delete;
private:
    InferProfiler &prof_;
    const char *name_;
    size_t layer_;
    bool active_;
    CudaTimer timer_;
};

} // namespace llaisys
