# Qwen3.8-Flash-Next on one DGX Spark

Serving a **126.0 GiB checkpoint on a machine with 121.63 GiB of memory** — at 41.5 tok/s on code,
across the full 262k context, scoring within noise of the checkpoint's published GSM8K.

Two small patches to SGLang, plus one flag most people won't find. Reproducible recipe,
microbenchmarks, and measured numbers below.

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

The idea is not mine — it is Qwen's, and it is in their [tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) (§2.3.2):

> Because embedding tables are sparsely accessed and deterministically addressed, they can be scaled
> with negligible additional per-token computation and **stored in off-accelerator storage**.

The whole layer is designed around it. §2.3.1: *"We place it at Layer 2, allowing host-memory
prefetching to overlap with the computation of the first layer."* What follows is an implementation
of that for SGLang on a unified-memory box, not a discovery.

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
--prefill-attention-backend triton      # trtllm_mha prefill is gated to SM100
--decode-attention-backend trtllm_mha   # decode explicitly allows SM120 — worth +32% on code
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

For long-context work, trade KV pool for headroom — see the stability warning below:

```
MEMFRAC=0.79 PREFILL=1024 CTX=262144 ./scripts/serve.sh
```

## Results

One GB10, 121.63 GiB unified memory, tp1, ctx 32k unless stated.

| | value |
|---|---|
| **GSM8K** (n=200, t=0.6, non-thinking, final config) | **192/200 = 96.0%** — checkpoint reference is 97.27% |
| decode, code EN (n=5, median) | **41.5 tok/s** (range 40.3–42.3) |
| decode, prose ES (n=5, median) | **22.8 tok/s** (range 21.2–25.5) |
| MTP acceptance | len 2.25–3.10 / 4, rate 0.42–0.70 |
| weights | 81.20 GB body + 0.71 GB draft |
| KV pool | 271,424 tokens, 16.41 GB free after capture |

**On the GSM8K number, and how far to trust it.** At n=200 the standard error is 1.39 pp, so 96.0%
sits inside the noise band around the 97.27% reference (the gap is 0.9 SE — not significant).

But the two numbers come from **different harnesses**: 97.27% is RadixArk's, measured with `sgl-eval`;
96.0% is [`verify.py`](verify.py) in this repo, with its own prompt and last-number extraction. Same
benchmark, different protocol, so this is an indicative comparison and not a like-for-like one. For
scale of how much protocol matters: Qwen's own tech report (Tab. 2) reports GSM8K **92.2** for this
model. Three numbers, three harnesses, one model.

So what this measurement actually establishes is narrow and worth stating plainly: it rules out the
silent failure mode of patch 1 — a PLE table served wrong from NVMe would not land here. It does not
prove zero degradation. At this sample size a 2–3 pp regression would be invisible, and against a
different harness the baseline itself moves by more than that.

An earlier run scored 59/60 = 98.3%, and that number was retired rather than kept: at n=60 one item is
worth 1.7 pp. It was the flattering read of a noisy sample. Note also that the two runs differ in more
than sample size — the n=200 run is on the final config (MTP + `trtllm_mha` decode) — so a clean A/B
would need n=200 without MTP, which has not been run.

**How the speed got there:** 13.1 tok/s on code at first boot → 31.5 with MTP → 41.5 once decode moved
to `trtllm_mha`. That's 3.2× with no change to the checkpoint. MTP helps code far more than prose, the
same asymmetry other speculative decoders show on this hardware — the drafter is right about
predictable text and wrong about prose.

---

## Long context: it works, and it will take the box down if you are careless

The full 262,144-token context runs. `--context-length 262144` needs no extra memory over 32k, because
the KV pool is sized in tokens and already holds more than one full context (273,536), and because
36 of the 48 layers are Gated DeltaNet whose recurrent state is **fixed size per request** and does not
grow with context. Measured at `--mem-fraction-static 0.85`, ctx 262144, one request at a time:

| Prompt tokens | Time to first token | Needle at midpoint retrieved |
|---:|---:|:--:|
| 8,103 | 12.1 s | yes |
| 32,104 | 38.8 s | yes |
| 128,104 | 144.3 s | yes |
| **240,104** | **331.9 s** | **yes** |

The needle is an invented fact ("the access code for the Argamasilla archive is …") inserted at the
halfway point of a Don Quijote excerpt — real, varied prose, not repeated text a prefix cache would
compress unfairly. QSA retrieves it at every length including a quarter of a million tokens.

**Prefix caching is what makes this usable.** Sending the same prompt a second time:

| Context | First pass | Cached |
|---:|---:|---:|
| 8k | 14.1 s | **0.2 s** |
| 128k | 183.0 s | **0.6 s** |
| 240k | 195.6 s | **1.7 s** |

Three minutes of prefill becomes under two seconds. For agent workloads that resend a large context
every turn, the corpus is paid for once per session rather than once per turn.

**Decode at long context**, measured over 300 generated tokens on a cache hit so the clock is decode
and not prefill:

| Context | Decode |
|---:|---:|
| 8k | 27.3 tok/s |
| 128k | 24.7 tok/s |
| 240k | **21.7 tok/s** |

About 20% of degradation across the whole range.

### The stability warning

Running a *sequence* of long prefills — four lengths up to 240k, flushing the prefix cache between
each so the numbers would be cold — **made the machine unresponsive**. Ping still answered and TCP
still accepted on every port, but no service completed a response and SSH died during banner
exchange: classic userspace starvation. It did not recover on its own.

At `--mem-fraction-static 0.85` SGLang takes ~103 GiB and leaves ~18 GiB for the OS, prefill
activations, and page cache for the 47.7 GiB PLE table. A 240k prefill on top of that is enough to
tip it. If you work at long context, give the box room:

- **`--mem-fraction-static 0.78`–`0.80`.** There is slack to give back: the KV pool holds 273,536
  tokens and one full context needs 262,144.
- **`--chunked-prefill-size 1024`** instead of 2048. Prefill activation memory scales with the chunk.
- **`--max-running-requests 1`** or 2 for long-context work. Each request costs 110 MB of recurrent
  state plus its KV.

**What I do not know is why the kernel could not reclaim its way out of it.** A 47.7 GiB clean,
file-backed mapping is exactly the kind of memory the page cache is supposed to drop under pressure.
Two observations that should fit together and don't:

- `mincore(2)` reported the whole table resident even immediately after `POSIX_FADV_DONTNEED` on a
  range of it, while `/proc/meminfo` reported 26 GiB of `Cached` total at the same moment. Those two
  cannot both be right, and I have not found which is wrong. (This is why there is no residency
  tooling in this repo — see above.)
- My first guess was that the loader *writes* the whole table on every boot, leaving 47.7 GiB of
  dirty pages that cannot be evicted until written back. **That guess does not survive its own data:**
  `Dirty` was 154 MB when I looked, so writeback had already happened and the pages were clean. The
  mechanism is still open.

Populating the file once and mapping it read-only afterwards remains the most interesting open item —
it would cut 2.5 minutes off every boot regardless, and if page state does turn out to matter it would
settle that too. But it is a hypothesis to test, not a fix I can claim.

### A methodology note on the prefill numbers

The time-to-first-token column above is trustworthy — those runs each carry a different needle, and the
clock is dominated by a long measurable phase. The *throughput* figures derived from them (roughly
670–890 tok/s) are **contaminated** and are not published as a headline: every length is a prefix of
the same corpus, so the longer runs reuse work the shorter ones cached. The tell is that one 240k run
reported 1,227 tok/s where a cold one gives ~700 — the most flattering number of the session was the
one that meant the least. A clean re-measurement with `/flush_cache` between lengths and disjoint
corpus offsets is the right way to do it; that run is what took the machine down, so it is unfinished.

## Gotchas the docs don't mention

- **`--attention-backend trtllm_mha` is refused, but `--decode-attention-backend trtllm_mha` is not.**
  The single flag sets both phases, and prefill is gated to SM100:

  ```
  ValueError: TRTLLM MHA backend for prefill is only supported on Blackwell GPUs (SM100).
  ```

  Read literally, that says the backend is unavailable. It isn't — the decode-side check in
  `server_args.py` lists `is_sm120_supported()` explicitly, so consumer Blackwell *is* a supported
  target for trtllm_mha decode. Splitting the phases gets you **+32% on code** (31.5 → 41.5 tok/s)
  and costs nothing. This one applies to any sm_120/121 box, model-independent.
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

## Why single-stream stops at ~42 tok/s

Not bandwidth. The decisive measurement is concurrency scaling on the same server:

| Concurrent requests | Aggregate | Per sequence | Scaling |
|---:|---:|---:|---|
| 1 | 42.8 tok/s | 42.8 | 1.00× |
| 2 | 53.2 | 29.1 | 1.25× |
| 4 | **95.2** | 27.3 | 2.23× |

The box delivers **95 tok/s** when given enough work. At C=1 it delivers 42.8. So single-stream is
**latency-bound, with roughly 2.2× of capacity sitting idle** — not starved of memory bandwidth, which
is what I assumed for most of a night before measuring it.

The way to use that headroom without concurrency is to verify more speculative tokens per forward.
C=4 with 4 draft tokens is 16 token-rows per forward and yields 95 tok/s; C=1 is 4 rows and yields
42.8. Eight rows at C=1 would interpolate to roughly 60.

**And that is exactly what is capped.** From `qwen_sparse_attn_backend.py`:

```
NotImplementedError: Qwen QSA requires speculative_num_draft_tokens <= the QSA compress ratio (4):
the pending index-key ring holds one group; got 8
```

with the reason in the comment right above it: *"The pending-group ring keys state by
`position % ratio`"*. The indexer's pending compressed-key ring has exactly `compress_ratio` slots, so
verifying more than 4 tokens at once makes two of them collide in the same slot and corrupts the keys
the sparse attention selects blocks with. That is a real correctness constraint of the implementation,
not a conservative guard — unlike the sm100 gate in patch 2, lifting this one would silently degrade
attention rather than fail loudly.

Lifting it means giving the ring multiple pending groups: `build_pending_ring_slots`, the ring
allocation in `graph_metadata.py`, the indexer kernel that reads it, and the CUDA-graph shapes. Four
files including a kernel, with a silent-corruption failure mode.

It is also probably not worth it, which is the part worth writing down. The payoff was estimated by
interpolating the concurrency curve, but 4 sequences x 4 draft tokens is not the same workload as
1 sequence x 16 - those four carry independent GDN states and KV, this one shares them. And acceptance
decays with draft depth: this setup accepts 2.77 of a possible 4, so draft tokens 5 through 8 would
land far below that. Iteration time scales sublinearly with rows (4x rows cost 1.8x time at C=4), so
eight rows at maybe 3.5 accepted works out around **48 tok/s** - a real gain, but not the 60 the
interpolation suggested, and not worth a correctness-risky rewrite to find out.

Single-stream on this checkpoint and this box looks like a ~42 tok/s problem.

## sm_121 is a second-class citizen, systematically

Patch 2 is not an isolated gap. Every one of these was hit on this box, with its exact error:

| What | Result on sm_121 |
|---|---|
| QSA trtllm-gen decode | gated to sm100; falls through to an FA4-cute path that will not compile (patch 2) |
| `--attention-backend trtllm_mha` | *"TRTLLM MHA backend for prefill is only supported on Blackwell GPUs (SM100)"* — decode is fine, the single flag sets both |
| `--moe-runner-backend flashinfer_trtllm` | *"cubin manifest contains no kernels runnable on sm121; ships cubins for sm100, sm103 and sm107"* |
| `--moe-runner-backend flashinfer_cutedsl` | *"No supported CUDA architectures found for major versions [10]"* |
| `--enable-torch-compile` | `NotImplementedError` during CUDA graph capture inside the model forward |

Of three flashinfer MoE backends, only `flashinfer_cutlass` runs here. Consumer Blackwell — RTX
50-series as much as GB10 — keeps landing on the generic path or on nothing at all.

## Kernel flags that changed nothing

Measured, n=10, medians, code prompt. All ranges overlap the baseline's 41.1–43.2:

| Configuration | tok/s |
|---|---:|
| baseline | 42.2 |
| `--speculative-attention-mode decode` + `--enable-linear-replayssm-spec` + draft on trtllm_mha | 43.6 |
| `--speculative-attention-mode decode` + `--fp4-gemm-backend flashinfer_cudnn` | 42.2 |
| `--num-continuous-decode-steps 2` | 41.5 |
| `--mamba-ssm-dtype bfloat16` (model declares float32) | 41.3 |

**No available kernel flag moves single-stream throughput measurably.** An earlier note in this repo
claimed `--speculative-attention-mode decode` was worth +17% in forward rate; that came from dividing
throughput by a noisy acceptance figure and does not survive direct comparison. Retracted.

Also already on, and worth knowing so nobody chases them: `SGLANG_ENABLE_QWEN4_PLE_FUSION` (default
true), `index_share_for_mtp_iteration` (default true), and the fused HC mix/combine kernels (this
model passes both of their conditions — batch ≤ 24 rows and `hc_count × hidden % 2048 == 0`).
`--enable-scattered-sconv` is a TP-multi-rank optimization and does nothing at `tp=1`.

## The ceiling does not move with acceptance either

Throughput here is `iterations/s x accepted tokens`, so the obvious remaining lever is acceptance —
and copy-heavy work is where speculative drafting should shine. Five tasks at C=1, ordered from most
to least "copy":

| Task | tok/s |
|---|---:|
| reproduce a code block verbatim | 42.3 |
| same block, one variable renamed | 42.8 |
| same block, type hints added | 42.4 |
| write a new function from scratch | 34.6 |
| free prose | 23.7 |

Acceptance did rise as predicted — median 3.33 against 2.77 on free generation, hitting the maximum
of 4.00 — and the verbatim reproduction was correct. **Throughput did not move.** Which means
iteration time grew in proportion: 66 ms per iteration at 2.77 accepted, 79 ms at 3.33.

A hypothesis for why, stated as one: something in the forward does not amortize over accepted tokens.
Attention and MoE do — one weight read serves all of them. A recurrence cannot, because `S_t` depends
on `S_{t-1}`, and **36 of the 48 layers are Gated DeltaNet**. If the GDN state has to be rolled
forward once per accepted token, accepting more tokens buys proportionally more sequential work and
throughput pins.

Halving the state width does not help, which is consistent: `--mamba-ssm-dtype bfloat16` against the
model's declared `float32` measured 41.3 tok/s. That rules out state *bandwidth* and leaves the
dependency chain, but it does not confirm it. The clean way to settle it would be profiling the GDN
kernel's share of iteration time, which is not done here.

## Where the speed actually goes

13.5 tok/s without speculation is about **41% of the memory-bandwidth roofline**, and the reason is
counter-intuitive: it isn't the experts.

The "6B active" are the NVFP4 part — roughly 1.2 GB read per token. But the ~3.5B **dense** parameters
(GDN, QSA, mHC, shared expert, embeddings) are still **BF16** and are read in full on every token:
~7 GB. So the small half of the model dominates the clock. At the Spark's ~273 GB/s that puts the
ceiling near 33 tok/s.

Two different kinds of headroom, then:

- **Implementation.** Moving decode to `trtllm_mha` already took code from 31.5 to 41.5 tok/s, which
  closed a good part of this gap. What's left untested: **more MTP steps.** Qwen's tech report (Tab. 4)
  measures a mean accepted length of **4.07 under four-step speculative decoding** (4.20 on GSM8K,
  4.26 on HumanEval). This recipe runs three steps and sees 2.25–3.10, so there is measurable room —
  and it's the paper's number, not a guess.
- **Another knob from the report** (§2.2, Inference Efficiency): Qwen keep the widened residual state
  in FP8, which *"halves the bytes moved for the residual state relative to BF16, with almost no loss
  in quality."* Since decode here is memory-bound, that goes straight to the clock — if SGLang exposes
  it for this architecture. Not checked.
- **Structural** (moves the roofline itself): quantizing the dense parameters to FP8 would take the
  ceiling from ~33 to ~55 tok/s. That's a requantization project, not a flag. Note that unsloth's GGUF
  conversions do quantize those parts, which is how `UD-Q4_K_XL` fits in 111.3 GB — at a measurable
  fidelity cost, and on a stack without prefix caching or MTP.

## What I have not measured

- **Clean cold prefill throughput.** See the methodology note above: the numbers exist but are
  contaminated by prefix reuse, and the uncontaminated re-run is the thing that took the box down.
- **Why the box could not reclaim.** The stability failure is reproducible and the mitigation works,
  but the mechanism is unexplained and my first explanation for it was wrong. See above.
- **A clean quality A/B.** The n=200 GSM8K run is on the final config; there is no matched n=200 run
  without MTP, so sample size and configuration changed together between the two quality runs.
- **Anything beyond GSM8K.** One benchmark, arithmetic-flavoured. The checkpoint's own card also
  publishes AIME26 (pass@1 98.75%), which would be a much stronger check and has not been run here.
- **Contexts beyond 32k**, and concurrency beyond 4 in-flight requests.
- **The other quantization path.** unsloth publishes GGUF conversions of this model where `UD-Q4_K_XL`
  (111.3 GB) fits a Spark natively, no patches — at 93.5% of full-precision fidelity by their own
  metric. Untested here. Their docs independently reach the same conclusion about the PLE table, which
  is worth reading: *"can be offloaded to SSD via mmap"*.

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
bench/test_long_prefill.py    prefill timing + needle retrieval at long context
bench/test_decode.py          reproducible decode benchmark (n samples, medians)
bench/test_concurrency.py     concurrency scaling -- is decode bandwidth-bound or latency-bound?
bench/test_copy_heavy.py      does higher draft acceptance raise throughput? (no)
```

## Page-cache residency, and a measurement I could not trust

[0xBakeer](https://github.com/0xBakeer/qwen38-flash-next-spark) make a point about this table that
applies here too:

> the n-gram table is 320M rows addressed by a 3-gram hash, so a workload almost never touches the
> same row twice early on — **it never warms naturally**, even when the table would fit in cache
> entirely.

Their fix is a tool that warms the table with one sequential read; their A/B measured +6% throughput
for it. I wrote the equivalent for this recipe and **withdrew it**: `mincore(2)` reported the full
47.7 GiB as resident even after explicitly evicting a range of it, which cannot be squared with the
26 GiB of `Cached` that `/proc/meminfo` reports at the same moment. The same tool behaves correctly on
a small file I control, so the bug is specific to this mapping and I have not found it. Rather than
ship a number I cannot reconcile, there is no residency tooling here yet.

One asymmetry worth flagging while that stays open: this recipe's loader *writes* the whole table at
startup, so every page passes through the cache on the way in — unlike an mmap'd read-only GGUF, which
only faults in what it touches. That may mean the table starts warm here for free. It is a plausible
story, not a measurement, which is exactly why the tool mattered.

## A note on method

The two mmap microbenchmarks were written **before** touching SGLang, which is why patch 1 worked on
the first boot. The attention microbenchmark was written *after* three nine-minute boots had learned
the same thing more slowly.

Kernel microbenchmark first, real boot second. `--load-format dummy` would be the obvious middle step
and it does not work here, for the reason above.

## Credits

- [Qwen](https://qwen.ai/blog?id=qwen3.8-flash-next) for the model, and for the
  [tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) — which says
  outright that these tables belong in off-accelerator storage (§2.3.2). Everything here follows from
  that sentence.
- [0xBakeer](https://github.com/0xBakeer/qwen38-flash-next-spark) for the llama.cpp recipe that lands
  on the same trick from a different stack, and for measuring what this repo had left open: full 262k
  context, prefill throughput, major-fault counts, and a table-warming A/B. their `warm_table.py` is the
  tool this repo still owes you.
- [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) for the NVFP4 checkpoint and,
  importantly, for publishing a GSM8K number for that exact checkpoint — without a reference score
  there is no way to tell "the model is like this" from "we broke it".
- [SGLang](https://github.com/sgl-project/sglang) for shipping `qwen4_exp` support on day zero.
- [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark) and
  [AEON-7](https://github.com/AEON-7/vllm-ultimate-dgx-spark) for the format — single-Spark recipes
  documented well enough to reproduce.

MIT.
