#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next NVFP4 on one DGX Spark.
#
# The checkpoint is 126.0 GiB and the Spark has 121.63 GiB of unified memory.
# It fits because of the two patches applied by prepare.sh -- see ../README.md.
#
# Env: MEMFRAC CTX PREFILL ATTN_PREFILL ATTN_DECODE SPEC LOADFMT NAME PORT PLE_DIR
set -euo pipefail

IMG=${IMG:-lmsysorg/sglang:qwen38flashnext}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"

PLE_DIR=${PLE_DIR:-$HOME/flashnext-ple}   # mmap backing store; sparse, persists
MEMFRAC=${MEMFRAC:-0.85}                  # 0.72 gives total_rest_memory < 0
CTX=${CTX:-32768}
PREFILL=${PREFILL:-2048}
# flashinfer hits a CUTLASS SM120 kernel that won't compile, so prefill is triton.
# Decode is a different story: --attention-backend trtllm_mha is refused because it
# sets BOTH phases and prefill is gated to SM100, but the decode-side check in
# server_args.py lists is_sm120_supported() explicitly. Splitting them is worth
# +32% on code (31.5 -> 41.5 tok/s).
ATTN_PREFILL=${ATTN_PREFILL:-triton}
ATTN_DECODE=${ATTN_DECODE:-trtllm_mha}
SPEC=${SPEC:-1}                           # 1 = MTP via NEXTN
LOADFMT=${LOADFMT:-auto}                  # NB: "dummy" OOMs with PLE offload, see README
NAME=${NAME:-flashnext}
PORT=${PORT:-30000}

for f in qwen4_exp.py qwen_sparse_attn_backend.py sm121_varlen.py kda_kernels path_qwen4_exp.txt path_qsa.txt path_sm121_varlen.txt path_kda_kernels.txt; do
  [ -f "$BUILD/$f" ] || { echo "missing $BUILD/$f — run ./scripts/prepare.sh first"; exit 1; }
done
mkdir -p "$PLE_DIR"
PATH_MODEL=$(cat "$BUILD/path_qwen4_exp.txt")
PATH_QSA=$(cat "$BUILD/path_qsa.txt")
PATH_SM121=$(cat "$BUILD/path_sm121_varlen.txt")
PATH_KDA=$(cat "$BUILD/path_kda_kernels.txt")

# MTP: 1 layer, already inside the checkpoint. Its 31 tensors are still BF16
# (RadixArk quantized only the routed experts), hence "unquant" for the draft:
# with modelopt_fp4 inherited from the body it would read them as NVFP4.
# PLE requires topk=1, which is exactly what NEXTN does.
SPEC_ARGS=()
if [ "$SPEC" = "1" ]; then
  SPEC_ARGS=(
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --speculative-draft-model-quantization unquant
  )
fi

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  --device nvidia.com/gpu=all --ipc host \
  -p "$PORT":30000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PLE_DIR":/ple \
  -v "$BUILD/qwen4_exp.py":"$PATH_MODEL":ro \
  -v "$BUILD/qwen_sparse_attn_backend.py":"$PATH_QSA":ro \
  -v "$BUILD/sm121_varlen.py":"$PATH_SM121":ro \
  -v "$BUILD/kda_kernels":"$PATH_KDA":ro \
  -e SGLANG_QWEN4_PLE_MMAP_DIR=/ple \
  "$IMG" \
  python3 -m sglang.launch_server \
    --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
    --served-model-name qwen38-flash-next \
    --host 0.0.0.0 --port 30000 \
    --load-format "$LOADFMT" \
    --tp-size 1 \
    --prefill-attention-backend "$ATTN_PREFILL" \
    --decode-attention-backend "$ATTN_DECODE" \
    --quantization modelopt_fp4 \
    --ple-offload-embedding \
    --language-only \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --mamba-radix-cache-strategy extra_buffer \
    --mem-fraction-static "$MEMFRAC" \
    --context-length "$CTX" \
    --chunked-prefill-size "$PREFILL" \
    --max-running-requests 4 \
    --allow-auto-truncate \
    --enable-metrics \
    "${SPEC_ARGS[@]}"

echo "started ($NAME, port $PORT, spec=$SPEC, load=$LOADFMT)"
echo "first boot takes ~9 min. logs: docker logs -f $NAME"
echo
echo "NOTE: --host 0.0.0.0 with no auth in front. Do not expose this to a network you don't trust."
