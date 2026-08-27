"""Decode on copy-heavy work: where speculative drafting should pay off most.

Throughput is `iterations/s x accepted tokens`. Iterations are pinned by the
hardware; the only thing that moves the result is acceptance, which depends on
how predictable the output is. Free prose accepts 0.42 here, code 0.70. Tasks
whose output reproduces the context -- editing, reformatting, quoting verbatim --
should approach the ceiling of 4.

Result on one GB10: acceptance does rise (median 3.33 vs 2.77, peaking at 4.00)
and throughput does not move. See the README.

    N=5 python3 bench/test_copy_heavy.py
"""
import json
import os
import statistics
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:30000/v1")
MODEL = os.environ.get("MODEL", "qwen38-flash-next")
N = int(os.environ.get("N", "5"))

CODIGO = '''def procesa_pedido(pedido, inventario, descuentos):
    total = 0.0
    for linea in pedido["lineas"]:
        sku = linea["sku"]
        if sku not in inventario:
            raise KeyError(f"sku desconocido: {sku}")
        disponible = inventario[sku]["stock"]
        if linea["cantidad"] > disponible:
            raise ValueError(f"stock insuficiente para {sku}")
        precio = inventario[sku]["precio"]
        subtotal = precio * linea["cantidad"]
        if sku in descuentos:
            subtotal *= (1.0 - descuentos[sku])
        total += subtotal
    return round(total, 2)'''

TAREAS = [
    ("copia literal",
     "Reproduce exactamente el siguiente codigo, sin cambiar nada:\n\n" + CODIGO),
    ("renombrado",
     "Devuelve este codigo con la variable `total` renombrada a `importe_total`. "
     "Todo lo demas identico:\n\n" + CODIGO),
    ("anadir type hints",
     "Devuelve este codigo anadiendo type hints a la firma. El cuerpo, identico:\n\n" + CODIGO),
    ("continuacion de codigo",
     "Escribe una funcion Python que valide un IBAN espanol. Solo el codigo."),
    ("prosa libre",
     "Explica en un parrafo que es la prescripcion de una sancion administrativa."),
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
    return (ct / dt if dt else 0.0), ct, r["choices"][0]["message"].get("content") or ""


print("tarea                  | mediana | rango           | tok")
print("-" * 62)
for etiqueta, p in TAREAS:
    una(p)  # calentamiento
    v = []
    ct = 0
    txt = ""
    for _ in range(N):
        s, c, t = una(p)
        v.append(s)
        ct = c
        txt = t
    print("%-22s | %6.1f  | %5.1f - %5.1f   | %d"
          % (etiqueta, statistics.median(v), min(v), max(v), ct))
    if "copia literal" in etiqueta:
        ok = "def procesa_pedido" in txt and "importe_total" not in txt
        print("     (reproduce el codigo: %s)" % ("si" if ok else "NO"))
