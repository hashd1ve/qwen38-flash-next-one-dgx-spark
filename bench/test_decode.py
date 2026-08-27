"""Reproducible decode benchmark. Same prompts, same shape, n=5, medians.

Discards a warmup request and reports median plus range, because MTP acceptance
varies and a single sample cannot tell a real gain from noise. Right after a boot
one warmup is not enough -- an n=5 run taken immediately after "fired up" read
32.9 tok/s where n=10 a few minutes later read 42.2 on the identical server.

    N=10 python3 bench/test_decode.py
"""
import json
import os
import statistics
import sys
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:30000/v1")
MODEL = os.environ.get("MODEL", "qwen38-flash-next")
N = int(os.environ.get("N", "5"))

PRUEBAS = [
    ("codigo EN", "Write a Python function that merges two sorted lists. Code only."),
    ("prosa ES", "Explica en tres frases que es la prescripcion de una sancion administrativa."),
]


def una(prompt, max_tokens=400):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.time() - t0
    ct = r["usage"]["completion_tokens"]
    return ct / dt if dt else 0.0, ct


print("etiqueta      | mediana | rango           | muestras")
print("-" * 62)
resumen = {}
for etiqueta, p in PRUEBAS:
    una(p)  # calentamiento, se descarta
    v = [una(p)[0] for _ in range(N)]
    resumen[etiqueta] = statistics.median(v)
    print("%-13s | %6.1f  | %5.1f - %5.1f   | %s"
          % (etiqueta, statistics.median(v), min(v), max(v),
             " ".join("%.1f" % x for x in v)))

print()
for k, v in resumen.items():
    print("RESULTADO %s: %.1f tok/s" % (k, v))
