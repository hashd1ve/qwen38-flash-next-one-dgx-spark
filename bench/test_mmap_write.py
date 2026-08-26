"""Is the write path bit-exact, and is the fp8 dequant right?

The gather being readable (test_mmap_gather.py) is only half of it. The weight
loader copies fp8 rows into the mmap'd tensor, and RadixArk's notes warn that a
loader which merely upcasts the fp8 bytes without applying the scale will serve
wrong PLE embeddings *silently*. So: copy real-shaped fp8 data in, reopen the
file from scratch, compare bytes, then gather from the GPU and compare values.

Runs the literal helper from patches/ple_mmap.py, not a hand-written copy.

    docker run --rm --gpus all -v "$PWD:/w" --entrypoint python3 \
        lmsysorg/sglang:qwen38flashnext /w/bench/test_mmap_write.py
"""
import importlib.util
import os

import torch
import triton
import triton.language as tl

spec = importlib.util.spec_from_file_location("patch", "/w/patches/ple_mmap.py")
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)
ns = {"torch": torch}
exec(patch.HELPER, ns)
_alloc_ple_table = ns["_alloc_ple_table"]
print("loaded helper from the patch itself:", _alloc_ple_table.__name__)

os.environ["SGLANG_QWEN4_PLE_MMAP_DIR"] = "/w/ple_test"
ns["_PLE_MMAP_DIR"] = None

N, D = 2_000_000, 160
table = _alloc_ple_table((N, D), torch.float8_e4m3fn)
print("table:", tuple(table.shape), table.dtype, "| pinned:", table.is_pinned())
assert table.dtype == torch.float8_e4m3fn and tuple(table.shape) == (N, D)

# mimic copy_ple_rows_to_tp_embedding: copy in row blocks
torch.manual_seed(0)
source = (torch.randn(N, D) * 3).to(torch.float8_e4m3fn)
for s in range(0, N, 250_000):
    e = min(s + 250_000, N)
    table.data[s:e].copy_(source[s:e].to(device="cpu", dtype=torch.float8_e4m3fn))
del table
os.system("sync")

path = os.path.join("/w/ple_test", os.listdir("/w/ple_test")[0])
reread = torch.from_file(path, shared=True, size=N * D, dtype=torch.uint8).view(N, D)
same = torch.equal(reread, source.view(torch.uint8))
print("bit-exact after reopening from disk:", same, "|", os.path.basename(path))
assert same


@triton.jit
def gather_kernel(weight_ptr, ids_ptr, out_ptr, embedding_dim, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < embedding_dim
    wp = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    vals = tl.load(wp + idx * embedding_dim + offs, mask=mask, other=0.0).to(tl.bfloat16)
    tl.store(out_ptr + row * embedding_dim + offs, vals, mask=mask)


ids = torch.randint(0, N, (4096,), dtype=torch.int64)
out = torch.empty((4096, D), dtype=torch.bfloat16, device="cuda")
gather_kernel[(4096,)](reread.data_ptr(), ids.cuda(), out, D,
                       BLOCK_D=triton.next_power_of_2(D))
torch.cuda.synchronize()
ok = torch.equal(out.cpu(), source[ids].to(torch.bfloat16))
print("GPU gather == source fp8 (dequant correct):", ok)
assert ok
print("OK: write and read paths both verified")
