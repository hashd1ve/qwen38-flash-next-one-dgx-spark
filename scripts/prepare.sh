#!/usr/bin/env bash
# Pull the image, extract the two files that need patching from it, patch them (and
# stage the sm121 QSA kernel that patch 2 imports),
# and verify the result by AST. Patching the image's own copy (rather than a
# vendored one) keeps this working if the PR moves under us.
set -euo pipefail

IMG=${IMG:-lmsysorg/sglang:qwen38flashnext}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
mkdir -p "$BUILD"

docker image inspect "$IMG" >/dev/null 2>&1 || docker pull "$IMG"

patch_one() {
  local module="$1" patcher="$2" out="$3" ref="$4"
  local path
  path=$(docker run --rm --entrypoint python3 "$IMG" -c \
    "import $module as m; print(m.__file__)" 2>/dev/null | tail -1)
  echo "$path" > "$BUILD/$ref"
  echo "  $module -> $path"

  local cid
  cid=$(docker create "$IMG")
  docker cp "$cid:$path" "$BUILD/$out"
  docker rm -f "$cid" > /dev/null

  python3 "$ROOT/patches/$patcher" "$BUILD/$out"
}

echo "extracting and patching:"
patch_one sglang.srt.models.qwen4_exp \
          ple_mmap.py qwen4_exp.py path_qwen4_exp.txt
patch_one sglang.srt.layers.attention.qwen_sparse_attn_backend \
          qsa_sm121_kda.py qwen_sparse_attn_backend.py path_qsa.txt
# patch 2 ships two kernels: the merged upstream KDA package (fast path, exact
# contract) and the 2026-08-28 Triton kernel as its fallback, both mounted.
cp "$ROOT/patches/qsa_sm121_varlen.py" "$BUILD/sm121_varlen.py"
echo "$(dirname "$(cat "$BUILD/path_qsa.txt")")/qsa/sm121_varlen.py" > "$BUILD/path_sm121_varlen.txt"
cp -R "$ROOT/patches/kda_kernels" "$BUILD/kda_kernels"
KERNELS_DIR=$(docker run --rm --entrypoint python3 "$IMG" -c \
  "import sglang.kernels as k; print(k.__path__[0])" 2>/dev/null | tail -1)
echo "$KERNELS_DIR/kda_kernels" > "$BUILD/path_kda_kernels.txt"
echo "  kda_kernels -> $KERNELS_DIR/kda_kernels"

echo "verifying:"
docker run --rm -v "$BUILD:/b" --entrypoint python3 "$IMG" -c "
import ast

src = open('/b/qwen4_exp.py').read()
tree = ast.parse(src)
pos = {n.name: n.lineno for n in tree.body if hasattr(n, 'name')}
assert '_alloc_ple_table' in pos, 'helper missing'
assert pos['_alloc_ple_table'] < pos['Qwen4ExpPinnedHostEmbedding']
cls = [n for n in tree.body if getattr(n, 'name', None) == 'Qwen4ExpPinnedHostEmbedding'][0]
body = ast.get_source_segment(src, cls)
assert '_alloc_ple_table(source_weight.shape' in body
assert 'pin_memory' not in body
print('  patch 1 (PLE mmap): ok')

src = open('/b/qwen_sparse_attn_backend.py').read()
ast.parse(src)
assert 'is_sm100_supported() or is_sm120_supported()' not in src, 'old patch 2 present'
assert 'kda_kernels.qwen38_qsa_sm121' in src
assert 'qsa.sm121_varlen' in src
ast.parse(open('/b/sm121_varlen.py').read())
ast.parse(open('/b/kda_kernels/qwen38_qsa_sm121/__init__.py').read())
ast.parse(open('/b/kda_kernels/qwen38_qsa_sm121/kernel.py').read())
print('  patch 2 (QSA sm121: KDA kernel + Triton fallback): ok')
"
echo "ready — now ./scripts/serve.sh"
