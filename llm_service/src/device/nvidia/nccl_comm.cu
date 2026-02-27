#include "nccl_comm.cuh"

namespace llaisys::device::nvidia {

NcclComm::~NcclComm() {
    for (int i = 0; i < ndev_; i++) {
        cudaSetDevice(device_ids_[i]);
        cudaStreamDestroy(streams_[i]);
        ncclCommDestroy(comms_[i]);
    }
}

void NcclComm::init(int ndev, const int *device_ids) {
    ndev_ = ndev;
    device_ids_.assign(device_ids, device_ids + ndev);
    comms_.resize(ndev);
    streams_.resize(ndev);

    NCCL_CHECK(ncclCommInitAll(comms_.data(), ndev, device_ids_.data()));

    for (int i = 0; i < ndev; i++) {
        cudaSetDevice(device_ids_[i]);
        cudaStreamCreate(&streams_[i]);
    }
}

void NcclComm::allReduceSum(void *buf, size_t count, ncclDataType_t dtype, int dev_idx) {
    NCCL_CHECK(ncclAllReduce(buf, buf, count, dtype, ncclSum, comms_[dev_idx], streams_[dev_idx]));
}

void NcclComm::allReduceSumBf16(void *buf, size_t count, int dev_idx) {
    allReduceSum(buf, count, ncclBfloat16, dev_idx);
}

void NcclComm::groupStart() {
    NCCL_CHECK(ncclGroupStart());
}

void NcclComm::groupEnd() {
    NCCL_CHECK(ncclGroupEnd());
}

void NcclComm::sync(int dev_idx) {
    cudaSetDevice(device_ids_[dev_idx]);
    cudaStreamSynchronize(streams_[dev_idx]);
}

void NcclComm::syncAll() {
    for (int i = 0; i < ndev_; i++) {
        sync(i);
    }
}

} // namespace llaisys::device::nvidia
