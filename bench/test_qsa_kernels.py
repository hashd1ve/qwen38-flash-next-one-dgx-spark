"""The two QSA decode paths, in isolation. No model load.

Answers in seconds what otherwise costs a 9-minute boot:
  1) reproduces the FA4-cute varlen failure (the one that kills startup on sm_121)
  2) says whether trtllm_batch_decode_with_kv_cache runs here

Shapes come from the real config: 24 Q heads, 2 KV heads, head_dim 256, page 64.

REQUIRES A FREE GPU. With the model loaded there is not enough memory left to
create a CUDA context, and both paths fail with a misleading OOM instead of
telling you anything about the kernels.

Run each path in its own process -- a failure in the first poisons the CUDA
context for the second:

    docker run --rm --gpus all -v "$PWD:/w" --entrypoint python3 \
        lmsysorg/sglang:qwen38flashnext /w/bench/test_qsa_kernels.py fa4
    docker run --rm --gpus all -v "$PWD:/w" --entrypoint python3 \
        lmsysorg/sglang:qwen38flashnext /w/bench/test_qsa_kernels.py trtllm
"""
import sys

import torch

ONLY = sys.argv[1] if len(sys.argv) > 1 else ""
print("capability:", torch.cuda.get_device_capability())

Hq, Hkv, D, PAGE = 24, 2, 256, 64
B, SEQ = 4, 512
dev, dt = "cuda", torch.bfloat16

if ONLY in ("", "fa4"):
    print("\n" + "=" * 60)
    print("1) FA4 cute varlen  (where sm_121 lands today)")
    print("=" * 60)
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func

        q = torch.randn(B, Hq, D, device=dev, dtype=dt)
        k = torch.randn(B * SEQ, Hkv, D, device=dev, dtype=dt)
        v = torch.randn(B * SEQ, Hkv, D, device=dev, dtype=dt)
        cu_q = torch.arange(0, B + 1, device=dev, dtype=torch.int32)
        cu_k = torch.arange(0, B + 1, device=dev, dtype=torch.int32) * SEQ
        out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, 1, SEQ, causal=True)
        torch.cuda.synchronize()
        print("   WORKS:", (out[0] if isinstance(out, tuple) else out).shape)
    except Exception as e:
        print("   FAILS:", type(e).__name__)
        print("  ", str(e).strip()[:600])

if ONLY in ("", "trtllm"):
    print("\n" + "=" * 60)
    print("2) trtllm_batch_decode_with_kv_cache  (what the gate discards)")
    print("=" * 60)
    try:
        import inspect

        from flashinfer.decode import trtllm_batch_decode_with_kv_cache

        print("   signature:", ", ".join(list(inspect.signature(
            trtllm_batch_decode_with_kv_cache).parameters)[:12]))
        npages = B * (SEQ // PAGE)
        q = torch.randn(B, Hq, D, device=dev, dtype=dt)
        kv = torch.randn(npages, 2, Hkv, PAGE, D, device=dev, dtype=dt)
        page_tbl = torch.arange(npages, device=dev, dtype=torch.int32).view(B, -1)
        seq_lens = torch.full((B,), SEQ, device=dev, dtype=torch.int32)
        wbuf = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
        out = trtllm_batch_decode_with_kv_cache(
            query=q, kv_cache=kv, workspace_buffer=wbuf,
            block_tables=page_tbl, seq_lens=seq_lens,
            max_seq_len=SEQ, bmm1_scale=D ** -0.5, bmm2_scale=1.0)
        torch.cuda.synchronize()
        print("   WORKS:", out.shape, out.dtype)
    except Exception as e:
        print("   FAILS:", type(e).__name__)
        print("  ", str(e).strip()[:600])
