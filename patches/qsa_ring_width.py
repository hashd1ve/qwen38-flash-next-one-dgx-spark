#!/usr/bin/env python3
"""EXPERIMENTAL: ensancha el anillo de claves pendientes de la QSA.

Motivo. En single-stream el decode esta limitado por latencia, no por ancho de
banda: la maquina da 95 tok/s agregados a C=4 y 42.8 a C=1. La unica forma de
llenar ese hueco sin concurrencia es verificar mas tokens de borrador por
forward, y eso lo topa el anillo del indexer en `compress_ratio` (4):

    Qwen QSA requires speculative_num_draft_tokens <= the QSA compress ratio (4):
    the pending index-key ring holds one group

El anillo direcciona por `req_pool_idx * ratio + position % ratio`, asi que con
8 tokens de borrador las posiciones p y p+4 caen en la misma ranura y se pisan
dentro del mismo kernel, antes de que la compresion del primer grupo pueda leer
sus miembros. Darle mas ranuras resuelve la colision.

Lo que NO cambia: `compress_ratio` sigue siendo el tamano del micro-bloque en
todos sus otros usos (block_topk, aritmetica de bloques, paginado comprimido).
Este parche solo separa la ANCHURA DEL ANILLO de ese valor, y solo en los tres
sitios que lo indexan o lo reservan. Todo lo demas queda intacto por
construccion, no por revision.

Coste en memoria: el buffer es (ring_slots, kv_heads, head_dim) con unos
cientos de ranuras. Doblarlo es despreciable.

RIESGO. Si esto esta mal, el indexer elige bloques equivocados y el modelo
empeora EN SILENCIO. Verificar siempre con la aguja a 240k (que ejercita
exactamente la seleccion de bloques) y con GSM8K, y no dar por bueno un
arranque limpio.

Se activa con SGLANG_QSA_RING_WIDTH=8. Sin la variable, comportamiento original.

Uso:
  python3 parche_ring.py pool    <ruta a mem_cache/qsa_kv_pool.py>
  python3 parche_ring.py meta    <ruta a layers/attention/qsa/metadata.py>
  python3 parche_ring.py backend <ruta a layers/attention/qwen_sparse_attn_backend.py>
"""
import sys

HELPER = '''

def _qsa_ring_width(compress_ratio: int) -> int:
    """Ranuras por peticion en el anillo de claves pendientes.

    Por defecto `compress_ratio`, que es el comportamiento original. Con
    SGLANG_QSA_RING_WIDTH se puede ensanchar para admitir mas tokens de
    borrador por forward; debe ser multiplo de compress_ratio para que
    `position % width` siga separando micro-bloques completos.
    """
    import os

    raw = os.environ.get("SGLANG_QSA_RING_WIDTH", "").strip()
    if not raw:
        return int(compress_ratio)
    width = int(raw)
    if width < compress_ratio or width % compress_ratio != 0:
        raise ValueError(
            "SGLANG_QSA_RING_WIDTH debe ser multiplo de compress_ratio "
            f"({compress_ratio}) y >= el; recibido {width}"
        )
    return width

'''

CAMBIOS = {
    # 1) la reserva del buffer
    "pool": [
        (
            "        ring_slots = self.qsa_num_request_slots * self.qsa_compress_ratio",
            "        ring_slots = self.qsa_num_request_slots * _qsa_ring_width(\n"
            "            self.qsa_compress_ratio\n"
            "        )",
        ),
    ],
    # 2) las dos formulas de ranura
    "meta": [
        (
            "    slots = requests * compress_ratio + positions % compress_ratio\n"
            "    if is_extend:\n"
            "        lengths = sequence_lengths.long()[rows]\n"
            "        pending = positions >= (lengths // compress_ratio) * compress_ratio\n"
            "        slots = torch.where(pending, slots, positions % compress_ratio)\n"
            "    return slots",
            "    width = _qsa_ring_width(compress_ratio)\n"
            "    slots = requests * width + positions % width\n"
            "    if is_extend:\n"
            "        lengths = sequence_lengths.long()[rows]\n"
            "        pending = positions >= (lengths // compress_ratio) * compress_ratio\n"
            "        slots = torch.where(pending, slots, positions % width)\n"
            "    return slots",
        ),
        (
            "    return requests[:, None] * compress_ratio + positions % compress_ratio",
            "    width = _qsa_ring_width(compress_ratio)\n"
            "    return requests[:, None] * width + positions % width",
        ),
    ],
    # 3) el guard
    "backend": [
        (
            "        if draft_tokens > self.compress_ratio:",
            "        if draft_tokens > _qsa_ring_width(self.compress_ratio):",
        ),
    ],
}

# donde insertar el helper en cada fichero (primera linea de nivel superior tras imports)
ANCLAS = {
    "pool": "class QSATokenToKVPool",
    "meta": "def build_pending_ring_slots(",
    "backend": "_TRTLLM_SPARSE_PAGE_SIZE = 64",
}


def main(cual: str, ruta: str) -> int:
    with open(ruta, encoding="utf-8") as f:
        src = f.read()

    if "_qsa_ring_width" in src:
        print("YA PARCHEADO:", ruta)
        return 0

    for viejo, nuevo in CAMBIOS[cual]:
        if src.count(viejo) != 1:
            print("ERROR: se esperaba 1 aparicion, encontradas %d, en %s"
                  % (src.count(viejo), ruta))
            print("  buscando:", viejo.split("\n")[0][:70])
            return 1
        src = src.replace(viejo, nuevo, 1)

    ancla = ANCLAS[cual]
    if src.count(ancla) < 1:
        print("ERROR: no se localizo el ancla %r en %s" % (ancla, ruta))
        return 1
    idx = src.index(ancla)
    src = src[:idx] + HELPER.lstrip("\n") + "\n" + src[idx:]

    with open(ruta, "w", encoding="utf-8") as f:
        f.write(src)
    print("PARCHEADO (%s): %s" % (cual, ruta))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
