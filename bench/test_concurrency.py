"""Diagnostic: what is actually limiting decode.

If the bottleneck is reading weights, those bytes are read ONCE per forward and
serve every sequence in the batch, so aggregate throughput should scale nearly
linearly with concurrency while per-sequence holds. If aggregate flattens early,
the limit is something else and there is headroom at batch 1.

On one GB10 this reads 42.8 / 53.2 / 95.2 tok/s aggregate at C=1/2/4 -- the box
has roughly 2.2x of capacity idle at C=1. See the README.

    python3 bench/test_concurrency.py
"""
import json
import os
import statistics
import threading
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:30000/v1")
MODEL = os.environ.get("MODEL", "qwen38-flash-next")
PROMPT = "Write a Python function that merges two sorted lists. Code only."


def una(res, i):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT + " Variant %d." % i}],
        "max_tokens": 300,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
        dt = time.time() - t0
        res[i] = (r["usage"]["completion_tokens"], dt)
    except Exception as e:
        res[i] = (0, 0.0)


print("concurrencia | agregado tok/s | por secuencia | escalado")
print("-" * 58)
base = None
for c in (1, 2, 4):
    # calentamiento
    r0 = {}
    una(r0, 0)

    res = {}
    hilos = [threading.Thread(target=una, args=(res, i)) for i in range(c)]
    t0 = time.time()
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    pared = time.time() - t0

    total_tok = sum(v[0] for v in res.values())
    agregado = total_tok / pared if pared else 0.0
    por_sec = statistics.median([v[0] / v[1] for v in res.values() if v[1] > 0] or [0])
    if base is None:
        base = agregado
    print("%11d  | %13.1f  | %12.1f  | %.2fx"
          % (c, agregado, por_sec, agregado / base if base else 0))
