#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rearrange_nvidia.cuh"
#endif

#include <cstring>

namespace llaisys::ops {

static inline ptrdiff_t compute_offset(const std::vector<size_t> &index,
                                       const std::vector<ptrdiff_t> &strides) {
    ptrdiff_t offset = 0;
    for (size_t i = 0; i < index.size(); i++) {
        offset += static_cast<ptrdiff_t>(index[i]) * strides[i];
    }
    return offset;
}

static inline bool increment_index(std::vector<size_t> &index,
                                   const std::vector<size_t> &shape) {
    for (size_t i = index.size(); i > 0; i--) {
        size_t dim = i - 1;
        index[dim]++;
        if (index[dim] < shape[dim]) {
            return true;  
        }
        index[dim] = 0; 
    }
    return false;  
}

template <typename T>
void rearrange_cpu(std::byte *out_ptr, const std::byte *in_ptr,
                   const std::vector<size_t> &shape,
                   const std::vector<ptrdiff_t> &out_strides,
                   const std::vector<ptrdiff_t> &in_strides) {
    T *out = reinterpret_cast<T *>(out_ptr);
    const T *in = reinterpret_cast<const T *>(in_ptr);

    size_t ndim = shape.size();
    if (ndim == 0) {
        // Scalar case
        out[0] = in[0];
        return;
    }

    std::vector<size_t> index(ndim, 0);
    do {
        ptrdiff_t out_offset = compute_offset(index, out_strides);
        ptrdiff_t in_offset = compute_offset(index, in_strides);
        out[out_offset] = in[in_offset];
    } while (increment_index(index, shape));
}

void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    auto shape = in->shape();
    auto out_strides = out->strides();
    auto in_strides = in->strides();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (in->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rearrange_cpu<float>(out->data(), in->data(), shape, out_strides, in_strides);
        case LLAISYS_DTYPE_F16:
            return rearrange_cpu<fp16_t>(out->data(), in->data(), shape, out_strides, in_strides);
        case LLAISYS_DTYPE_BF16:
            return rearrange_cpu<bf16_t>(out->data(), in->data(), shape, out_strides, in_strides);
        case LLAISYS_DTYPE_I64:
            return rearrange_cpu<int64_t>(out->data(), in->data(), shape, out_strides, in_strides);
        case LLAISYS_DTYPE_I32:
            return rearrange_cpu<int32_t>(out->data(), in->data(), shape, out_strides, in_strides);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
        }
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        size_t numel = 1;
        for (auto s : shape) numel *= s;
        return nvidia::rearrange(out->data(), in->data(), shape, out_strides, in_strides, in->dtype(), numel);
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
