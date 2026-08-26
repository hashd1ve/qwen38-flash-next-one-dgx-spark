"""Verify the server. Content before speed.

The check that matters for patch 1 is GSM8K: RadixArk publishes 97.27% for this
exact checkpoint (BF16 reference band 97.12-97.50). If the NVMe-backed PLE table
were being served wrong, the score would fall visibly. A model that got slower
is obvious; a model that got dumber is not.

    python3 verify.py                # N=60 by default, ~20 min
    N=200 python3 verify.py          # tighter interval
"""
import json
import os
import re
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:30000/v1")
N = int(os.environ.get("N", "60"))
MODEL = os.environ.get("MODEL", "qwen38-flash-next")


def chat(msgs, max_tokens=1024, thinking=True, temp=0.6, top_p=0.95):
    body = {
        "model": MODEL,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    dt = time.time() - t0
    m = r["choices"][0]["message"]
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    return m.get("content") or "", reasoning, r.get("usage", {}), dt


print("=" * 62)
print("1. CONTENT AND DECODE SPEED")
print("=" * 62)
for label, prompt in [
    ("prose", "Explain in three sentences what a statute of limitations is."),
    ("code ", "Write a Python function that merges two sorted lists. Code only."),
]:
    txt, _, us, dt = chat([{"role": "user", "content": prompt}], max_tokens=512, thinking=False)
    ct = us.get("completion_tokens", 0)
    print("\n[%s] %d tok in %.1fs = %.1f tok/s" % (label, ct, dt, ct / dt if dt else 0.0))
    print("  " + (txt[:200].replace("\n", "\n  ") or "*** EMPTY ***"))
    # An image that "serves" at speed while emitting empty tokens is a real
    # failure mode. Check the bytes, not the throughput.
    assert len(txt.strip()) > 40, "empty or trivial output for " + label

print()
print("=" * 62)
print("2. THINKING CONTROL")
print("=" * 62)
for th in (True, False):
    txt, reasoning, us, _ = chat(
        [{"role": "user", "content": "What is 17*23? Answer with the number only."}],
        max_tokens=2048, thinking=th,
    )
    print("  enable_thinking=%-5s -> reasoning_content: %4d chars | content: %r"
          % (str(th), len(reasoning), txt.strip()[:60]))
print("  (without --reasoning-parser qwen3 the thinking text lands in content)")

print()
print("=" * 62)
print("3. QUALITY — GSM8K n=%d (reference: 97.27%%)" % N)
print("=" * 62)
import pandas as pd
from huggingface_hub import hf_hub_download

f = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
df = pd.read_parquet(f).head(N)
good = bad = 0
t0 = time.time()
for i, row in df.iterrows():
    ref = row["answer"].split("####")[-1].strip().replace(",", "")
    txt, _, _, _ = chat(
        [{"role": "user", "content": row["question"] + "\n\nEnd with: #### <number>"}],
        max_tokens=2048, thinking=False, temp=0.6,
    )
    nums = re.findall(r"-?\d+\.?\d*", txt.replace(",", ""))
    got = nums[-1] if nums else None
    try:
        ok = got is not None and abs(float(got) - float(ref)) < 1e-4
    except ValueError:
        ok = False
    good += ok
    bad += not ok
    if not ok and bad <= 3:
        print("  miss #%d: expected %s, got %s" % (i, ref, got))
    if (i + 1) % 10 == 0:
        print("  ... %d/%d: %d correct = %.1f%%" % (i + 1, N, good, 100.0 * good / (i + 1)))

print()
print("  GSM8K %d/%d = %.1f%%  (reference 97.27%%)  in %.0fs"
      % (good, N, 100.0 * good / N, time.time() - t0))
