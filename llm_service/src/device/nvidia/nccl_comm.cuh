#pragma once

#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <string>

#define NCCL_CHECK(call)                                                       \
    do {                                                                       \
        ncclResult_t res = (call);                                             \
        if (res != ncclSuccess) {                                              \
            throw std::runtime_error(std::string("NCCL error: ") +             \
                                     ncclGetErrorString(res));                 \
        }                                                                      \
    } while (0)

namespace llaisys::device::nvidia {

class NcclComm {
public:
    NcclComm() = default;
    ~NcclComm();

    void init(int ndev, const int *device_ids);

    void allReduceSum(void *buf, size_t count, ncclDataType_t dtype, int dev_idx);
    void allReduceSumBf16(void *buf, size_t count, int dev_idx);

    void groupStart();
    void groupEnd();
    void sync(int dev_idx);
    void syncAll();

    int size() const { return ndev_; }
    int deviceId(int idx) const { return device_ids_[idx]; }
    cudaStream_t stream(int idx) const { return streams_[idx]; }

private:
    int ndev_ = 0;
    std::vector<int> device_ids_;
    std::vector<ncclComm_t> comms_;
    std::vector<cudaStream_t> streams_;
};

} // namespace llaisys::device::nvidia
