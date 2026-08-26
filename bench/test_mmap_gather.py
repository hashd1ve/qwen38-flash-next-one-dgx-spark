"""Can a Triton kernel read a non-pinned, file-backed mmap on this hardware?

This is the question patch 1 rests on, and it takes 30 seconds to answer instead
of a 9-minute boot. Replicates SGLang's own PLE gather kernel (int64 pointer ->
fp8 -> bf16) against an mmap'd file, with the page cache dropped by hand so the
cold path is real. Row width 160 B is the model's head_dim_per_ngram.

    docker run --rm --gpus all -v "$PWD:/w" --entrypoint python3 \
        lmsysorg/sglang:qwen38flashnext /w/bench/test_mmap_gather.py
"""
import ctypes
import os
import time

import torch
import triton
import triton.language as tl

cudart = ctypes.CDLL("libcudart.so")


def attr(a):
    v = ctypes.c_int()
    rc = cudart.cudaDeviceGetAttribute(ctypes.byref(v), ctypes.c_int(a), ctypes.c_int(0))
    return v.value if rc == 0 else "err%d" % rc


print("capability:", torch.cuda.get_device_capability())
print("PageableMemoryAccess               (88) =", attr(88))
print("PageableMemAccessUsesHostPageTables(100) =", attr(100))


@triton.jit
def gather_kernel(weight_ptr, ids_ptr, out_ptr, embedding_dim, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < embedding_dim
    wp = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    vals = tl.load(wp + idx * embedding_dim + offs, mask=mask, other=0.0).to(tl.bfloat16)
    tl.store(out_ptr + row * embedding_dim + offs, vals, mask=mask)


D = 160                 # head_dim_per_ngram
N = 20_000_000          # 3.2 GB fp8 on disk (the real table is 320M rows)
PATH = os.environ.get("TABLE", "/w/ple_test.bin")
NBYTES = N * D

if not os.path.exists(PATH) or os.path.getsize(PATH) != NBYTES:
    print("creating backing file, %.1f GB ..." % (NBYTES / 1e9))
    t0 = time.time()
    with open(PATH, "wb") as f:
        f.truncate(NBYTES)
    w = torch.from_file(PATH, shared=True, size=NBYTES, dtype=torch.uint8).view(N, D)
    for s in range(0, N, 2_000_000):
        e = min(s + 2_000_000, N)
        rows = torch.arange(s, e, dtype=torch.int64)
        w[s:e] = ((rows % 126) + 1).to(torch.uint8).unsqueeze(1).expand(-1, D)
    del w
    os.system("sync")
    print("  written in %.1fs" % (time.time() - t0))


def drop_cache():
    os.system("sync")
    fd = os.open(PATH, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


drop_cache()
w8 = torch.from_file(PATH, shared=True, size=NBYTES, dtype=torch.uint8).view(N, D)
print("table:", tuple(w8.shape), "| is_pinned:", w8.is_pinned(), "| ptr:", hex(w8.data_ptr()))

torch.manual_seed(0)
BD = triton.next_power_of_2(D)
for label, nrows in (("decode  (16 rows)", 16), ("prefill (65,536 rows)", 65536)):
    ids = torch.randint(0, N, (nrows,), dtype=torch.int64)
    ids_gpu = ids.cuda()
    out = torch.empty((nrows, D), dtype=torch.bfloat16, device="cuda")

    gather_kernel[(1,)](w8.data_ptr(), ids_gpu, out, D, BLOCK_D=BD)  # JIT compile apart
    torch.cuda.synchronize()

    drop_cache()
    t0 = time.time()
    gather_kernel[(nrows,)](w8.data_ptr(), ids_gpu, out, D, BLOCK_D=BD)
    torch.cuda.synchronize()
    cold = time.time() - t0

    expected = ((ids % 126) + 1).to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    correct = bool(torch.equal(expected, out[:, 0].to(torch.float32).cpu()))

    t0 = time.time()
    gather_kernel[(nrows,)](w8.data_ptr(), ids_gpu, out, D, BLOCK_D=BD)
    torch.cuda.synchronize()
    warm = time.time() - t0

    print("  %-22s correct=%s  cold=%.2f ms  warm=%.2f ms"
          % (label, correct, cold * 1000, warm * 1000))
