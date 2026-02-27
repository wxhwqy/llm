#include "nvidia_resource.cuh"
#include <stdexcept>
#include <string>
#include <unordered_map>

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error(                                          \
                std::string("cuBLAS error: ") + std::to_string(status));       \
        }                                                                      \
    } while (0)

namespace llaisys::device::nvidia {

Resource::Resource(int device_id)
    : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    cudaSetDevice(device_id);
    CUBLAS_CHECK(cublasCreate(&_cublas_handle));
}

Resource::~Resource() {
    cublasDestroy(_cublas_handle);
    if (_tile_buf) cudaFree(_tile_buf);
}

void *Resource::getTileBuffer(size_t size_bytes) {
    if (size_bytes > _tile_buf_size) {
        if (_tile_buf) cudaFree(_tile_buf);
        cudaMalloc(&_tile_buf, size_bytes);
        _tile_buf_size = size_bytes;
    }
    return _tile_buf;
}

Resource &Resource::get() {
    static thread_local std::unordered_map<int, Resource *> instances;
    int device_id = 0;
    cudaGetDevice(&device_id);
    auto it = instances.find(device_id);
    if (it == instances.end()) {
        auto *r = new Resource(device_id);
        instances[device_id] = r;
        return *r;
    }
    return *it->second;
}

} // namespace llaisys::device::nvidia
