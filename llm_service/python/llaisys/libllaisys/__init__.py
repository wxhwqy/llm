import os
import sys
import ctypes
from pathlib import Path

from .runtime import load_runtime
from .runtime import LlaisysRuntimeAPI
from .llaisys_types import llaisysDeviceType_t, DeviceType
from .llaisys_types import llaisysDataType_t, DataType
from .llaisys_types import llaisysMemcpyKind_t, MemcpyKind
from .llaisys_types import llaisysStream_t
from .tensor import llaisysTensor_t
from .tensor import load_tensor
from .ops import load_ops
from .models import (
    load_models,
    LlaisysQwen3Meta, LlaisysQwen3Weights, LlaisysQwen3FP8Linear, LlaisysQwen3Model_p,
)


def _preload_nccl():
    """Preload NCCL shared library so libllaisys can resolve NCCL symbols."""
    if not sys.platform.startswith("linux"):
        return
    nccl_paths = [
        os.path.join(sys.prefix, "lib/python" + f"{sys.version_info.major}.{sys.version_info.minor}",
                     "site-packages/nvidia/nccl/lib/libnccl.so.2"),
        "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
        "/usr/local/cuda/lib64/libnccl.so.2",
    ]
    for p in nccl_paths:
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


def load_shared_library():
    _preload_nccl()
    lib_dir = Path(__file__).parent

    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "llaisys.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    lib_path = os.path.join(lib_dir, libname)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Shared library not found: {lib_path}")

    return ctypes.CDLL(str(lib_path))


LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)
load_models(LIB_LLAISYS)


__all__ = [
    "LIB_LLAISYS",
    "LlaisysRuntimeAPI",
    "llaisysStream_t",
    "llaisysTensor_t",
    "llaisysDataType_t",
    "DataType",
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "llaisysStream_t",
    "LlaisysQwen3Meta",
    "LlaisysQwen3Weights",
    "LlaisysQwen3FP8Linear",
    "LlaisysQwen3Model_p",
]
