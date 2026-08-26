#!/usr/bin/env python3
"""Back the Qwen3.8-Flash-Next PLE n-gram table with an NVMe mmap.

SGLang's PR #36497 puts the table (47.7 GiB in fp8) in pinned host memory. On a
host with separate system RAM that frees the equivalent VRAM; on a DGX Spark the
memory is unified, so it comes out of the same 121.63 GiB pool and the model
(126.0 GiB of weights) will not boot.

GB10 reports cudaDevAttrPageableMemoryAccessUsesHostPageTables == 1, and the
Triton gather kernel has been verified to read correctly from a non-pinned,
file-backed mmap (bench/test_mmap_gather.py). So only the backing store changes:
the kernel, the prefetch stream and the CUDA graphs are untouched.

Usage:  python3 ple_mmap.py <path to qwen4_exp.py>
"""
import re
import sys

HELPER = '''

_PLE_MMAP_DIR = None


def _alloc_ple_table(shape, dtype):
    """Backing store for the PLE n-gram table.

    With SGLANG_QWEN4_PLE_MMAP_DIR set, the table is a file-backed mmap on disk
    instead of pinned host RAM. On coherent CPU-GPU parts (GB10) the gather
    kernel dereferences that pageable host pointer directly, so the table never
    has to be resident. Without the variable, original behaviour.
    """
    import logging
    import os

    global _PLE_MMAP_DIR
    if _PLE_MMAP_DIR is None:
        _PLE_MMAP_DIR = os.environ.get("SGLANG_QWEN4_PLE_MMAP_DIR", "").strip()
    if not _PLE_MMAP_DIR:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    numel = 1
    for d in shape:
        numel *= int(d)
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    os.makedirs(_PLE_MMAP_DIR, exist_ok=True)
    path = os.path.join(_PLE_MMAP_DIR, "ple_table_%d_%d.bin" % (numel, nbytes))
    if not os.path.exists(path) or os.path.getsize(path) != nbytes:
        with open(path, "wb") as f:
            f.truncate(nbytes)
    logging.getLogger(__name__).info(
        "PLE table -> mmap %s (%.1f GiB, dtype=%s)", path, nbytes / 2**30, dtype
    )
    storage = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)

    # MADV_RANDOM: the table is purely random access (16 rows of 160 B per
    # token). Without it the kernel reads its whole readahead window around each
    # row. Measured cold: 1.4 MB of disk per token, ~560x more than is used.
    # Cheap in decode; a prefill chunk touches tens of thousands of rows.
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        MADV_RANDOM = 1
        rc = libc.madvise(
            ctypes.c_void_p(storage.data_ptr()),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(MADV_RANDOM),
        )
        logging.getLogger(__name__).info(
            "PLE table: madvise(MADV_RANDOM) %s", "ok" if rc == 0 else "failed"
        )
    except Exception as exc:  # not critical: affects I/O only, never correctness
        logging.getLogger(__name__).warning("PLE table: madvise not applied (%s)", exc)

    return storage.view(dtype).view(*shape)

'''

ORIG = re.compile(
    r"""cpu_weight\s*=\s*nn\.Parameter\(\s*
        torch\.empty\(\s*
            source_weight\.shape\s*,\s*
            dtype\s*=\s*source_weight\.dtype\s*,\s*
            device\s*=\s*["']cpu["']\s*,\s*
            pin_memory\s*=\s*True\s*,?\s*
        \)\s*,\s*
        requires_grad\s*=\s*False\s*,?\s*
    \)""",
    re.VERBOSE,
)

NEW = """cpu_weight = nn.Parameter(
            _alloc_ple_table(source_weight.shape, source_weight.dtype),
            requires_grad=False,
        )"""

ANCHOR = "class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):"


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        src = f.read()

    if "_alloc_ple_table" in src:
        print("ALREADY PATCHED:", path)
        return 0

    n = len(ORIG.findall(src))
    if n != 1:
        print("ERROR: expected 1 pinned allocation, found %d" % n)
        return 1
    src = ORIG.sub(NEW, src)

    if src.count(ANCHOR) != 1:
        print("ERROR: could not locate Qwen4ExpPinnedHostEmbedding")
        return 1
    src = src.replace(ANCHOR, HELPER.lstrip("\n") + "\n" + ANCHOR, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
