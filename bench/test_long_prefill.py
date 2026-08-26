"""Long-prefill timing with a needle, up to the full 262k context.

Measures, per prompt length: time to first token (the prefill), and whether an
invented fact buried at the midpoint is retrieved. The corpus should be real,
varied prose -- repeated text would let the prefix cache compress it unfairly.

WARNING, read the stability section of the README first: a sequence of long
prefills at --mem-fraction-static 0.85 made the machine unresponsive. Give the
box headroom (MEMFRAC=0.79 PREFILL=1024) before running the long lengths.

CAVEAT: successive lengths here are prefixes of the same corpus, so the longer
runs reuse cache from the shorter ones and the derived throughput is optimistic.
The time-to-first-token figures are sound; the tok/s ones are not. Flush
/flush_cache between lengths and use disjoint corpus offsets for clean numbers.

    CORPUS=/path/to/long_text.txt python3 bench/test_long_prefill.py 8000 32000
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:30000/v1")
MODEL = os.environ.get("MODEL", "qwen38-flash-next")
CORPUS = os.environ.get("CORPUS", "/w/quijote.txt")

texto = open(CORPUS, encoding="utf-8", errors="ignore").read()
# quitar la licencia de Gutenberg de la cabecera
i = texto.find("PRIMERA PARTE")
if i > 0:
    texto = texto[i:]
print("corpus: %.1f MB" % (len(texto) / 1e6))

# tokenizador real del modelo, para saber cuantos tokens pedimos de verdad
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("RadixArk/Qwen3.8-Flash-Next-NVFP4", trust_remote_code=True)
print("tokenizador cargado")


def corta_a_tokens(txt, n):
    """Recorta el texto a aproximadamente n tokens."""
    # ~3.5 caracteres por token en espanol; ajustar iterando
    aprox = txt[: int(n * 4.5)]
    ids = tok(aprox, add_special_tokens=False)["input_ids"]
    while len(ids) < n and len(aprox) < len(txt):
        aprox = txt[: int(len(aprox) * 1.2)]
        ids = tok(aprox, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:n])


def prueba(objetivo_tokens, idx):
    aguja = "El codigo de acceso al archivo de Argamasilla es %d-ZULU-%d." % (
        700 + idx, 30 + idx)
    cuerpo = corta_a_tokens(texto, objetivo_tokens)
    mitad = len(cuerpo) // 2
    cuerpo = cuerpo[:mitad] + "\n\n" + aguja + "\n\n" + cuerpo[mitad:]
    prompt = (
        "A continuacion tienes un fragmento de El Quijote. En algun punto del texto "
        "hay una frase que NO pertenece a la obra: dice cual es el codigo de acceso "
        "al archivo de Argamasilla.\n\n" + cuerpo +
        "\n\nPregunta: cual es el codigo de acceso al archivo de Argamasilla? "
        "Responde solo con el codigo.")

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 40,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    ttft = None
    salida = ""
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for linea in r:
            linea = linea.decode("utf-8").strip()
            if not linea.startswith("data: "):
                continue
            datos = linea[6:]
            if datos == "[DONE]":
                break
            o = json.loads(datos)
            if o.get("usage"):
                usage = o["usage"]
            for ch in o.get("choices", []):
                trozo = (ch.get("delta") or {}).get("content") or ""
                if trozo and ttft is None:
                    ttft = time.time() - t0
                salida += trozo
    total = time.time() - t0

    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    cached = usage.get("prompt_tokens_details") or {}
    ok = ("ZULU-%d" % (30 + idx)) in salida.replace(" ", "")
    print("  %8d tok prompt | TTFT %7.1fs = %6.0f tok/s prefill | decode %5.1f tok/s "
          "| aguja %s | %r"
          % (pt, ttft or total, pt / (ttft or total),
             (ct - 1) / (total - ttft) if ttft and total > ttft else 0.0,
             "SI" if ok else "NO", salida.strip()[:40]))
    return pt, ttft, ok


if __name__ == "__main__":
    objetivos = [int(x) for x in (sys.argv[1:] or ["8000", "32000", "128000", "240000"])]
    print("\nlongitud objetivo -> resultado")
    for i, n in enumerate(objetivos):
        try:
            prueba(n, i)
        except Exception as e:
            print("  %8d tok: FALLO %s: %s" % (n, type(e).__name__, str(e)[:120]))
