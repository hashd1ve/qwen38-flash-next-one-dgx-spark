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
          qsa_sm121_triton.py qwen_sparse_attn_backend.py path_qsa.txt
# the Triton kernel behind patch 2 (sglang#36845): mounted next to the backend
cp "$ROOT/patches/qsa_sm121_varlen.py" "$BUILD/sm121_varlen.py"
echo "$(dirname "$(cat "$BUILD/path_qsa.txt")")/qsa/sm121_varlen.py" > "$BUILD/path_sm121_varlen.txt"

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
assert 'qsa.sm121_varlen' in src
ast.parse(open('/b/sm121_varlen.py').read())
print('  patch 2 (QSA sm121 Triton fallback): ok')
"
echo "ready — now ./scripts/serve.sh"
