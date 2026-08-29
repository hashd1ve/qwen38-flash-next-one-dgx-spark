#!/usr/bin/env python3
"""QSA sparse decode on sm_121 (GB10): Triton varlen fallback from sglang#36845.

This REPLACES the original patch 2 of this recipe (qsa_trtllm_sm120.py), which
widened `_resolve_trtllm_sparse_decode` to sm_12x. That was wrong. On sm_121
flashinfer does not run trtllm-gen for that call (no sm12x cubins): with
`backend="auto"` it routes to XQA, and XQA silently corrupts long-context decode
on GB10. Measured upstream with real weights on two independent DGX Spark
setups (sglang#36556, comment by dpolistwm; sglang#36806, comment by BBuf):
runs of token id 0 (`!`) in 1/4 requests at 120k prompt tokens, 2/4 at 190k,
4/4 at 210k -- HTTP 200 and /health fine throughout. It is stochastic and only
reachable with more than `indexer_budget` (2048) tokens of history, which is why
short benchmarks, GSM8K and single needle runs never caught it.

Upstream fixed it in two steps on the qwen4-main-squashed branch:
  - sglang#36806 (2026-08-28): the trtllm gate is `is_sm100_supported() or
    is_sm120()` (exact 12.0); sm_121 goes back to the varlen fallback.
  - sglang#36845 (2026-08-28, BBuf): on sm_121 that fallback is a small Triton
    online-softmax kernel (`qsa/sm121_varlen.py`: one query per sequence,
    device-side cu_seqlens, CUDA-graph safe), because the pip FA4-cute path does
    not compile for QSA's packed shape on GB10.

What this patcher does to the image's pristine `qwen_sparse_attn_backend.py`:
leaves the trtllm gate exactly as upstream shipped it (sm100 only -- the image
predates #36806 and has no `is_sm120()`, which makes no difference on a GB10)
and inserts the #36845 hunk into `_resolve_flash_attn_varlen_func`. The kernel
itself ships alongside as patches/qsa_sm121_varlen.py and serve.sh mounts it.

Validated on one DGX Spark (2026-08-29): exact needle retrieval 4/4 at each of
120k, 190k and 210k prompt tokens; decode 30-48 tok/s after those prompts;
13-case code suite unchanged.

Usage:  python3 qsa_sm121_triton.py <path to qwen_sparse_attn_backend.py>
"""
import sys

GATE_WIDENED = "if not (is_sm100_supported() or is_sm120_supported()):"

ANCHOR = """    try:
        from flash_attn import flash_attn_varlen_func
"""

HUNK = """    from sglang.srt.utils import is_sm121

    if is_sm121():
        from sglang.srt.layers.attention.qsa.sm121_varlen import (
            qsa_sm121_varlen_attention,
        )

        return qsa_sm121_varlen_attention
""" + ANCHOR


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "qsa.sm121_varlen" in src:
        print("ALREADY PATCHED:", path)
        return 0
    if GATE_WIDENED in src:
        print("ERROR: the trtllm gate was widened (old patch 2). Re-extract the pristine file.")
        return 1
    n = src.count(ANCHOR)
    if n != 1:
        print(f"ERROR: expected 1 varlen fallback anchor, found {n}")
        return 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(ANCHOR, HUNK, 1))
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
