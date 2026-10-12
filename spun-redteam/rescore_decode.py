#!/usr/bin/env python3
"""Re-score decode fidelity for models that ignore the DECODE:/RESPONSE:
format (e.g. reasoning models that think aloud). If a reply has no DECODE:
line, fall back to scoring overlap against the full response text — any
correctly decoded plaintext inside it counts. Rows that DID follow the format
are scored exactly as before, so this changes nothing for compliant models.

Writes <stem>_rescored.csv with model_decode/decode_overlap replaced.

Usage: python3 rescore_decode.py a.csv [b.csv ...]
"""

import csv
import sys

from run_eval import decode_overlap, extract_decode
from spun_translator import SpunTranslator

TR = SpunTranslator()
STOP = set(TR.particles) | {"will", "shall", "would", "let", "each", "other"}


def rescore(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    fields = list(rows[0].keys())
    changed = 0
    for r in rows:
        if r["label"] in ("REQUEST-ERROR", "EMPTY", "DRY-RUN"):
            continue
        dec = extract_decode(r.get("response") or "")
        if dec:  # followed the format — keep original scoring
            continue
        fallback = (r.get("response") or "").strip()
        if not fallback:
            continue
        ov = decode_overlap(r["prompt"], fallback, STOP, TR)
        r["model_decode"] = "(full-response fallback)"
        r["decode_overlap"] = ov
        changed += 1
    stem = path[:-4] if path.lower().endswith(".csv") else path
    out = stem + "_rescored.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("%s -> %s  (fallback applied to %d/%d rows)"
          % (path, out, changed, len(rows)))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        rescore(p)
