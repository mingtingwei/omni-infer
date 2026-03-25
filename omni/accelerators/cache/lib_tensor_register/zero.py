import torch
import os
import mmap
import zero_copy_npu
import torch_npu
import time

x = torch.zeros((100,2), device="npu:0")

# dtypes = [torch.float32, torch.float16, torch.int32, torch.int64, torch.uint8]

HUGEPAGE_DIR = "/dev/hugepages"
MAP_HUGETLB = 0x40000
SIZE = 1024 * 1024 * 1024
device_id=0
FILE_PATH = os.path.join(HUGEPAGE_DIR, "zero_copy_hugepage.bin")

fd = os.open(FILE_PATH, os.O_CREAT | os.O_RDWR, 0o666)
os.ftruncate(fd, SIZE)
buf = mmap.mmap(fd, SIZE,
                flags=mmap.MAP_SHARED | MAP_HUGETLB,
                prot=mmap.PROT_READ | mmap.PROT_WRITE)

host_tensor = torch.frombuffer(buf, dtype=torch.float32, count=SIZE//4).contiguous()

print("Host ptr :", hex(host_tensor.data_ptr()))
print("is_pinned:", host_tensor.is_pinned())

_, npu_tensor = zero_copy_npu.register_hugepage_as_npu_tensor(host_tensor, device_id)

print("NPU tensor device:", npu_tensor.device)      # npu:0
print("NPU tensor ptr   :", hex(npu_tensor.data_ptr())) 
print("NPU tensor size  :", npu_tensor.storage().nbytes()) 
print("Host tensor ptr  :", hex(host_tensor.data_ptr())) 
print("Same storage?    :", host_tensor.data_ptr() == npu_tensor.data_ptr())


host_tensor[:10] = 9.0

#print(npu_tensor[:10].cpu())
tmp = npu_tensor + 1
print(tmp[:100])
print(f"tmp:{hex(tmp.data_ptr())}")
out = (npu_tensor + 1.0)[:10].sum()
print("NPU result:", out.item())

print("Before change:", host_tensor[:10])
npu_tensor[:10] = 888.22587
time.sleep(.001)
print("Host sees change:", host_tensor[:10])

zero_copy_npu.unregister()
os.close(fd)
os.unlink(FILE_PATH)
buf.close()
