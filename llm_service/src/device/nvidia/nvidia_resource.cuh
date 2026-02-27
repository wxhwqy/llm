#pragma once

#include "../device_resource.hpp"
#include <cublas_v2.h>
#include <cstddef>

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();

    cublasHandle_t cublasHandle() const { return _cublas_handle; }

    // Lazy-allocated tile buffer for tiled FP8 dequant+GEMM.
    // Grows to fit max needed size. Eliminates large per-model dequant buffers.
    void *getTileBuffer(size_t size_bytes);

    static Resource &get();

private:
    cublasHandle_t _cublas_handle;
    void *_tile_buf = nullptr;
    size_t _tile_buf_size = 0;
};
} // namespace llaisys::device::nvidia
