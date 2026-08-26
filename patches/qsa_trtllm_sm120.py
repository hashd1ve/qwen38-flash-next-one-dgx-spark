#!/usr/bin/env python3
"""Enable the QSA trtllm-gen decode kernel on sm_120/121 (consumer Blackwell).

`_resolve_trtllm_sparse_decode` drops the kernel when `is_sm100_supported()` is
false. GB10 is Blackwell but reports (12, 1), so it exits there and falls back to
the "packed varlen fallback", which needs FA2 (not in the SGLang image) or the
FA4 cute interface -- and that one does not compile on SM120:

    MLIRError: expects `coord` and shape of view are weakly congruent,
    at flash_attn/cute/flash_fwd.py:393 (epilogue)

Net result on sm_121: QSA has no working decode path at all.

flashinfer 0.6.17 does expose trtllm_batch_decode_with_kv_cache on this device,
so the gate is widened to sm_120/121. If the kernel did not support the
architecture it would fail with its own explicit error, which is a better place
to fail than inside another backend's MLIR compilation.

This is not DGX Spark specific: any consumer Blackwell (RTX 50-series included)
takes the same path.

Usage:  python3 qsa_trtllm_sm120.py <path to qwen_sparse_attn_backend.py>
"""
import sys

OLD = """    from sglang.srt.utils import is_sm100_supported

    if not is_sm100_supported():
        return None"""

NEW = """    from sglang.srt.utils import is_sm100_supported, is_sm120_supported

    # sm_120/121 (consumer Blackwell: GB10 / RTX 50-series) is Blackwell too and
    # flashinfer ships the trtllm-gen decode kernel for it. The original gate
    # excludes it, which forces the FA4-cute varlen fallback -- and that one
    # fails to compile on SM120.
    if not (is_sm100_supported() or is_sm120_supported()):
        return None"""


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        src = f.read()

    if "is_sm120_supported()" in src and "consumer Blackwell" in src:
        print("ALREADY PATCHED:", path)
        return 0

    n = src.count(OLD)
    if n != 1:
        print("ERROR: expected 1 occurrence of the gate, found %d" % n)
        return 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW, 1))
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
