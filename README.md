# Qwen3.8-Flash-Next on one DGX Spark

Serving a **126.0 GiB checkpoint on a machine with 121.63 GiB of memory**, at 31.5 tok/s, with no quality loss.

Two small patches to SGLang. Reproducible recipe, microbenchmarks, and measured numbers below.

> Day-zero work, 2026-08-26 — the model, the SGLang support PR and this repo are all the same day old.
> Everything here is measured on one GB10, not projected. Where I'm extrapolating, I say so.

---

## The problem

Qwen3.8-Flash-Next is 125B total / **6B active** + **51B of n-gram embeddings (PLE)** + 4B MTP. The only
checkpoint that gets close to a single Spark is [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4),
and it quantizes **only the routed experts**:

| Component | GB | GiB |
|---|---:|---:|
| Routed experts, NVFP4 (384 shards) | 68.0 | 63.3 |
| **PLE n-gram table, fp8** (10 shards) | 51.2 | **47.7** |
| BF16 (attn / GDN / mHC / vision / MTP / lm_head) | 16.0 | 14.9 |
| **Total** | **135.3** | **126.0** |

A DGX Spark has **121.63 GiB**. That's 4.4 GiB short before any KV cache, activations or CUDA context.

### Why the built-in offload doesn't help here

Every engine ships the same escape hatch, and on unified memory none of them work:

| Engine | Flag | Where the table goes |
|---|---|---|
| SGLang | `--ple-offload-embedding` | pinned host RAM |
| vLLM | `VLLM_PLE_CPU_OFFLOAD=1` | host RAM ("at least 51 GB plus runtime headroom") |
| llama.cpp | `-ot "ple_ngram_embd=CPU"` | host RAM |

On a B300, host RAM is *different memory* and you free 47.7 GiB of VRAM. On a Spark, **host RAM and GPU
memory are the same 121.63 GiB pool** — moving the table between them frees nothing. vLLM's official
recipe asks for TP2 minimum on GB300; the SGLang cookbook lists H200/B200/B300/GB300 and GB10 is not
on it.

---

## Patch 1 — the PLE table lives on NVMe

The 47.7 GiB PLE table is not a weight matrix you multiply. It's a **lookup table**: 20M rows of 160
bytes, and a token reads 16 of them. There is exactly one PLE layer (`ple_layer_ids: [2]`), so one
gather per forward pass.

GB10 reports **`cudaDevAttrPageableMemoryAccessUsesHostPageTables == 1`** — the GPU resolves host
virtual addresses through the host page tables, so a CUDA kernel can dereference a pointer into a
file-backed `mmap` and the OS services the faults. I verified this against SGLang's own Triton gather
kernel, with the page cache dropped by hand, *before* touching SGLang ([`bench/test_mmap_gather.py`](bench/test_mmap_gather.py)):

| | cold (page cache dropped) | warm |
|---|---:|---:|
| decode, 16 rows | 3.58 ms | 0.12 ms |
| prefill, 65,536 rows | 3,865 ms | 6.92 ms |

So the patch is three lines in `Qwen4ExpPinnedHostEmbedding.__init__` — swap `torch.empty(...,
pin_memory=True)` for a `torch.from_file(shared=True)`:

```diff
 cpu_weight = nn.Parameter(
-    torch.empty(source_weight.shape, dtype=source_weight.dtype,
-                device="cpu", pin_memory=True),
+    _alloc_ple_table(source_weight.shape, source_weight.dtype),
     requires_grad=False,
 )
```

**Nothing downstream changes.** The Triton gather kernel, the prefetch stream and the CUDA graphs all
still receive a host pointer — they don't care that it's now backed by a file. `cuda graph: True` in the
scheduler confirms graph capture still works.

Measured cost in service: **under 3% of wall clock.** 138 KB/token of disk reads against the 2.5 KB
actually used — that's 4 KB page granularity, not a bandwidth problem (20 MB/s on an NVMe). The patch
also sets `madvise(MADV_RANDOM)`, which matters more in long prefill than in decode.

The backing file is **sparse and persists across restarts** (deterministic name), so it only occupies
what has been written.

**Correctness is verified, not assumed** — see the GSM8K number below. A loader that only reinterprets
the fp8 bytes without applying the scale would serve wrong embeddings silently; [`bench/test_mmap_write.py`](bench/test_mmap_write.py)
checks the write path is bit-exact and the dequant matches.

## Patch 2 — QSA has no decode kernel on sm_121

`is_sm100_supported()` requires `major == 10`. GB10 is `(12, 1)`. Two independent sites gate on it and
together they dead-end:

1. **`arg_groups/overrides.py::_qwen3_5_hybrid_overrides`** returns `{}`, so `attention_backend` never
   receives the family default (`triton`) and falls through to the global default — `flashinfer`.
2. **`qwen_sparse_attn_backend.py::_resolve_trtllm_sparse_decode`** returns `None`, so QSA falls back to
   its packed varlen path, which needs FA2 (not in the image) or the FA4 cute interface.

FA4 cute then fails to *compile*:

```
MLIRError: expects `coord` and shape of view are weakly congruent, but got
'!cute.layout<"(?,?):(?{i64 div=8},1)">', '!cute.coord<"(_,_,?)">'
  flash_attn/cute/flash_fwd.py:393, in epilogue
```

It does select an SM120 path (`flash_attncuteflash_fwd_sm120FlashAttentionForwardSm120`) — a rank-2
layout indexed with a rank-3 coordinate. Net result: **no working QSA decode path at all** on sm_121.

Setting `--attention-backend triton` alone is *not* enough, because the second gate is independent of
that flag. Widening it is:

```diff
-    if not is_sm100_supported():
+    if not (is_sm100_supported() or is_sm120_supported()):
         return None
```

flashinfer 0.6.17 does ship a working `trtllm_batch_decode_with_kv_cache` on this device.

**This is not Spark-specific.** Any consumer Blackwell (sm_120/121) hits it — RTX 50-series included.
Reported upstream: [sgl-project/sglang#36497](https://github.com/sgl-project/sglang/pull/36497#issuecomment-5427832100).

---

## Quickstart

Needs: a DGX Spark (or another GB10), Docker, ~140 GB of free disk, and membership in the `docker` group.

```bash
git clone https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark
cd qwen38-flash-next-one-dgx-spark

# 1) weights — 135 GB, ~20 min at 100 MB/s
./scripts/download.sh

# 2) pull the image, extract the two files from it, patch them, verify by AST
./scripts/prepare.sh

# 3) serve on :30000
./scripts/serve.sh
```

First boot is ~9 minutes: 8.5 min of weight loading (CPU-bound on dequant, ~182 MB/s from disk) plus
pool allocation and graph capture. Roughly 2.5 of those minutes are the PLE table being written to its
backing file; subsequent boots rewrite the same bytes.

Then:

```bash
python3 verify.py     # content -> thinking control -> GSM8K, in that order
```

---

## The flags that matter, and why

```
--attention-backend triton              # flashinfer routes to a CUTLASS SM120 kernel that won't compile
--quantization modelopt_fp4
--ple-offload-embedding                 # with patch 1, "offload" means NVMe, not pinned RAM
--language-only
--mamba-radix-cache-strategy extra_buffer   # required: page_size=64 is forced for compressed QSA
--mem-fraction-static 0.85              # 0.72 dies with total_rest_memory=-1.00 GB
--reasoning-parser qwen3 --tool-call-parser qwen3_coder
--speculative-algorithm NEXTN --speculative-num-steps 3
--speculative-eagle-topk 1              # PLE requires topk=1
--speculative-num-draft-tokens 4
--speculative-draft-model-quantization unquant   # the 31 MTP tensors are BF16 in an NVFP4 checkpoint
```

## Results

One GB10, 121.63 GiB unified memory, tp1, ctx 32k.

| | value |
|---|---|
| **GSM8K** (n=60, t=0.6, non-thinking) | **59/60 = 98.3%** — checkpoint reference is 97.27% |
| decode, code EN | 13.1 tok/s → **31.5** with MTP |
| decode, prose ES | 13.6 tok/s → **19.5** with MTP |
| MTP acceptance | len 2.58–2.92 / 4, rate 0.53–0.64 |
| weights | 80.46 GB body + 1.38 GB draft |
| KV pool | 275,456 tokens, 15.45 GB free after capture |

n=60 gives roughly ±2 pp, so 98.3% is indistinguishable from the 97.27% reference — the point is that
**serving the PLE table from NVMe does not degrade the model**. That was the silent failure mode of
patch 1, and it's the number that rules it out.

MTP helps code far more than prose (2.40× vs 1.43×) — the same asymmetry other speculative decoders
show on this hardware: the drafter is right about predictable text and wrong about prose.

---

## Gotchas the docs don't mention

- **`--mem-fraction-static 0.72` will not boot.** `total_rest_memory=-1.00 GB`, dying in
  `_handle_max_mamba_cache` with `mamba_cache_per_req=110.11 MB`. Needs 0.85. Note the ceiling is
  artificial — the physical memory was free the whole time.
- **`reasoning_parser` is auto-detected but not applied.** Startup logs `Auto-detected template
  features: ... reasoning_parser=qwen3` while `server_args` shows `reasoning_parser=None`. Without
  passing it explicitly, thinking text lands in `content` instead of `reasoning_content`.
- **`--load-format dummy` is unusable with this model.** Dummy init materializes the 47.7 GiB PLE table
  in RAM and the OOM killer takes the scheduler. It removes the obvious fast smoke-test path.
- **`nvidia-smi` reports `[N/A]` for memory used on GB10.** Use `free`. SGLang falls back to
  `torch.cuda.mem_get_info()` on its own and reports 124546 MiB.
- **The MTP weights are BF16 inside an NVFP4 checkpoint** — without `unquant` the draft inherits
  `modelopt_fp4` from the body.
- PLE forbids two-batch overlap and NGRAM speculation, and requires `topk=1`.

## Where the speed actually goes

13.5 tok/s without speculation is about **41% of the memory-bandwidth roofline**, and the reason is
counter-intuitive: it isn't the experts.

The "6B active" are the NVFP4 part — roughly 1.2 GB read per token. But the ~3.5B **dense** parameters
(GDN, QSA, mHC, shared expert, embeddings) are still **BF16** and are read in full on every token:
~7 GB. So the small half of the model dominates the clock. At the Spark's ~273 GB/s that puts the
ceiling near 33 tok/s.

Two different kinds of headroom, then:

- **Implementation** (~2.4× to the roofline): `trtllm_mha` is also allowed on sm120 and I never tried
  it; `--speculative-num-steps` above 3 is untested with an acceptance length of 2.6/4.
- **Structural** (moves the roofline itself): quantizing the dense parameters to FP8 would take the
  ceiling from ~33 to ~55 tok/s. That's a requantization project, not a flag.

## What I have not measured

- **End-to-end with long prompts.** All decode numbers here are pure decode. With a large corpus in
  front, prefill dominates, and that's also where the NVMe-backed PLE would cost the most — the cold
  prefill microbenchmark says 3.9 s worst case for 65k rows with an empty page cache. Treat the
  sub-3% overhead figure as a decode result.
- **GSM8K with MTP enabled.** Speculative decoding should be lossless by construction; the 98.3% was
  measured without it.
- **Contexts beyond 32k**, and concurrency beyond 4 in-flight requests.

## Layout

```
patches/ple_mmap.py           patch 1 — applied to models/qwen4_exp.py
patches/qsa_trtllm_sm120.py   patch 2 — applied to layers/attention/qwen_sparse_attn_backend.py
scripts/download.sh           fetch the NVFP4 checkpoint
scripts/prepare.sh            extract from the image, patch, verify by AST
scripts/serve.sh              launch
verify.py                     content -> thinking control -> GSM8K
bench/test_mmap_gather.py     can a Triton kernel read an mmap'd file on this hardware?
bench/test_mmap_write.py      is the write path bit-exact and the fp8 dequant right?
bench/test_qsa_kernels.py     the two QSA decode paths, in isolation (needs a free GPU)
```

## A note on method

The two mmap microbenchmarks were written **before** touching SGLang, which is why patch 1 worked on
the first boot. The attention microbenchmark was written *after* three nine-minute boots had learned
the same thing more slowly.

Kernel microbenchmark first, real boot second. `--load-format dummy` would be the obvious middle step
and it does not work here, for the reason above.

## Credits

- [Qwen](https://qwen.ai/blog?id=qwen3.8-flash-next) for the model and the architecture writeup.
- [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) for the NVFP4 checkpoint and,
  importantly, for publishing a GSM8K number for that exact checkpoint — without a reference score
  there is no way to tell "the model is like this" from "we broke it".
- [SGLang](https://github.com/sgl-project/sglang) for shipping `qwen4_exp` support on day zero.
- [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark) and
  [AEON-7](https://github.com/AEON-7/vllm-ultimate-dgx-spark) for the format — single-Spark recipes
  documented well enough to reproduce.

MIT.
