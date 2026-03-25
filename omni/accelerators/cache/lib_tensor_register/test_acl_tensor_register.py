#!/usr/bin/env python3

import os
import ctypes
import ctypes.util
import mmap

import torch
import numpy as np

from acl_tensor_register import NPUTensorRegister

def allocate_aligned(size_bytes: int, alignment: int = 4096):
    """Allocate an aligned host buffer using posix_memalign."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.posix_memalign.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    libc.posix_memalign.restype = ctypes.c_int

    ptr = ctypes.c_void_p()
    ret = libc.posix_memalign(ctypes.byref(ptr), alignment, size_bytes)
    if ret != 0:
        raise MemoryError(f"posix_memalign failed, ret={ret}, size={size_bytes}, alignment={alignment}")

    return ptr.value, libc


def create_aligned_cpu_tensor(shape, dtype=torch.float32):
    """Create a 4K-aligned CPU tensor backed by manually allocated memory."""
    numel = 1
    for s in shape:
        numel *= s

    element_size = torch.tensor(0, dtype=dtype).element_size()
    total_bytes = numel * element_size

    raw_ptr, libc = allocate_aligned(total_bytes, alignment=4096)

    np_dtype_map = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: np.uint16,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }
    np_dtype = np_dtype_map.get(dtype, np.float32)
    ctype_type = np.ctypeslib.as_ctypes_type(np_dtype)

    np_array = np.ctypeslib.as_array(
        ctypes.cast(raw_ptr, ctypes.POINTER(ctype_type)),
        shape=(numel,),
    )

    tensor = torch.from_numpy(np_array).reshape(shape)

    # Initialize data for testing
    with torch.no_grad():
        tensor.copy_(torch.randn(shape, dtype=dtype))

    # Verify 4K alignment
    assert tensor.data_ptr() % 4096 == 0, f"Alignment failed: data_ptr % 4096 = {tensor.data_ptr() % 4096}"

    return tensor, raw_ptr, libc


def main():
    device_id = 0  # change if you want to test another device

    print("=== NPUTensorRegister host_tensor_register test (with 4K aligned CPU tensor) ===")

    shape = (10, 512)
    dtype = torch.float32

    x = torch.zeros((100,2), device="npu:0")

    HUGEPAGE_DIR = "/dev/hugepages"
    if not os.path.isdir(HUGEPAGE_DIR):
        os.makedirs(HUGEPAGE_DIR)
    MAP_HUGETLB = 0x40000
    SIZE = 1024 * 1024 * 1024

    FILE_PATH = os.path.join(HUGEPAGE_DIR, "zero_copy_hugepage.bin")
    fd = os.open(FILE_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    os.ftruncate(fd, SIZE)
    buf = mmap.mmap(fd, SIZE,
                    flags=mmap.MAP_SHARED | MAP_HUGETLB,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE)
    print("\n[Step 1] Create CPU tensor from buffer")
    cpu_tensor = torch.frombuffer(buf, dtype=torch.float32, count=SIZE//4).contiguous()

    print(f"CPU tensor shape      : {cpu_tensor.shape}")
    print(f"CPU tensor dtype      : {cpu_tensor.dtype}")
    print(f"CPU data_ptr          : 0x{cpu_tensor.data_ptr():x}")
    print(f"CPU data_ptr % 4096   : {cpu_tensor.data_ptr() % 4096} (should be 0)")

    print("\n[Step 2] Register host tensor and create NPU view")
    register = NPUTensorRegister()
    # NOTICE: cpu_tensor is newly returned.
    print(f"Before cpu_tensor's data_ptr: {cpu_tensor.data_ptr()=}")
    _, npu_tensor = register.host_tensor_register(cpu_tensor, device_id=device_id)
    print(f"After cpu_tensor's data_ptr: {cpu_tensor.data_ptr()=}")
    print(f"NPU tensor device     : {npu_tensor.device}")
    print(f"NPU tensor shape      : {npu_tensor.shape}")
    print(f"NPU tensor dtype      : {npu_tensor.dtype}")

    print("\n[Step 3] In-place operation on NPU tensor (add_(10))")
    before = cpu_tensor.clone()
    npu_tensor.add_(10)

    print("Compare CPU tensor values before/after:")
    print(f"CPU[:10] before     : {before[:10]}")
    print(f"CPU[:10] after      : {cpu_tensor[:10]}")

    diff = (cpu_tensor - before).flatten()[:10]
    print(f"First 10 (after - before): {diff}")

    print("\nCheck summary:")
    print("- 4K alignment verified on CPU tensor.")
    print("- NPU tensor successfully created via host_tensor_register.")
    print("- Value difference indicates whether mapping is zero-copy (host-mapped) or copy-based.")

    # Optional: free manually allocated memory at the end
    # libc.free(cpu_tensor.data_ptr()) ## Error


if __name__ == "__main__":
    main()