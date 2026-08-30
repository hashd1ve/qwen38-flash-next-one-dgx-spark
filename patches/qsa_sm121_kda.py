#!/usr/bin/env python3
"""QSA sparse decode on sm_121: the merged upstream kernel (KDA), with a Triton fallback.

sglang#36845 was merged on 2026-08-30 with a different kernel than its 2026-08-28
revision: a KDA-1.5 agent-optimized implementation (Codex + Kimi K3) living in
`sglang/kernels/kda_kernels/qwen38_qsa_sm121/`. On a GB10 it measures 4-5x the
2026-08-28 Triton kernel at decode batch 1-4 (28 us vs 154 at batch 1) with
identical numerics, and it is what upstream ships going forward. Its contract is
exact — BF16, head dim 256, 24Q/2KV (TP1) or 12Q/1KV (TP2), batch <= 128,
selected KV <= 2055 — so this recipe keeps the 2026-08-28 Triton kernel
(patches/qsa_sm121_varlen.py) as the fallback for anything outside it.

This patcher edits the image's pristine `qwen_sparse_attn_backend.py`: the
trtllm gate stays as shipped (sm100 only; the XQA route it would open on sm_121
corrupts long context — see the README), and `_resolve_flash_attn_varlen_func`
gets a route that calls the KDA kernel inside its contract and the Triton
fallback otherwise. The KDA package itself is vendored verbatim under
patches/kda_kernels/ and mounted by serve.sh.

Validated on one DGX Spark (2026-08-30): exact needle retrieval 9/9 at
120k/190k/210k prompt tokens; decode after those prompts 46-92 tok/s (the
2026-08-28 kernel: 30-48); short-context decode unchanged.

Usage:  python3 qsa_sm121_kda.py <path to qwen_sparse_attn_backend.py>
"""
import sys

GATE_WIDENED = "if not (is_sm100_supported() or is_sm120_supported()):"

ANCHOR = """    try:
        from flash_attn import flash_attn_varlen_func
"""

HUNK = """    from sglang.srt.utils import is_sm121

    if is_sm121():
        from sglang.kernels.kda_kernels.qwen38_qsa_sm121 import (
            can_use_qwen38_qsa_sm121,
            qwen38_qsa_sm121,
        )
        from sglang.srt.layers.attention.qsa.sm121_varlen import (
            qsa_sm121_varlen_attention,
        )

        def _qsa_sm121_kda_varlen(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=1, max_seqlen_k=0,
            softmax_scale=1.0, causal=True, **_,
        ):
            if max_seqlen_q == 1 and can_use_qwen38_qsa_sm121(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_k
            ):
                return qwen38_qsa_sm121(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_k, softmax_scale
                )
            return qsa_sm121_varlen_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale, causal=causal,
            )

        return _qsa_sm121_kda_varlen
""" + ANCHOR


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "kda_kernels.qwen38_qsa_sm121" in src:
        print("ALREADY PATCHED:", path)
        return 0
    if GATE_WIDENED in src:
        print("ERROR: the widened trtllm gate is present (deprecated patch). Re-extract the pristine file.")
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
