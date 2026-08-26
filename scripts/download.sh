#!/usr/bin/env bash
# Fetch the NVFP4 checkpoint (135.3 GB). Network and disk only: no --gpus.
# ~20 min at 100 MB/s. Needs ~140 GB free.
set -euo pipefail

IMG=${IMG:-lmsysorg/sglang:qwen38flashnext}
REPO=${REPO:-RadixArk/Qwen3.8-Flash-Next-NVFP4}
LOG=${LOG:-$HOME/flashnext-download.log}

echo "downloading $REPO -> ~/.cache/huggingface (log: $LOG)"
docker run --rm \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint python3 "$IMG" -c "
from huggingface_hub import snapshot_download
r = '$REPO'
print('>>> ' + r, flush=True)
print('    -> ' + snapshot_download(r, max_workers=8), flush=True)
print('DONE', flush=True)
" 2>&1 | tee -a "$LOG"
