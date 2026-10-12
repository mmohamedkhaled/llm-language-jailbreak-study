#!/usr/bin/env python3
"""Summarize run_eval.py result CSVs: label counts, Wilson 95% CI on the
refusal rate, decode-fidelity stats, and the valid-subset (overlap-filtered)
refusal rate. Usage: python3 summarize_results.py a.csv [b.csv ...]"""

import csv
import math
import sys

Z = 1.96


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return center - half, center + half


def summarize(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if "label" not in (rows[0].keys() if rows else []):
        print("== %s ==  (no 'label' column; not a run_eval output, skipped)\n" % path)
        return
    sent = [r for r in rows if r["label"] not in ("DRY-RUN",)]
    err = [r for r in sent if r["label"] in ("REQUEST-ERROR", "EMPTY")]
    scored = [r for r in sent if r["label"] not in ("REQUEST-ERROR", "EMPTY")]
    n = len(scored)
    counts = {}
    for r in scored:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    k = counts.get("REFUSAL", 0)
    lo, hi = wilson(k, n)
    print("== %s ==" % path)
    print("  n=%d (excluded infra: %d)" % (n, len(err)))
    for lab, c in sorted(counts.items()):
        print("    %-18s %3d" % (lab, c))
    print("  refusal rate %.2f  [95%% CI %.2f-%.2f]" % (k / n if n else 0, lo, hi))

    ovs = [float(r["decode_overlap"]) for r in scored if r.get("decode_overlap")]
    if ovs:
        valid = [r for r in scored if r.get("decode_overlap")
                 and float(r["decode_overlap"]) >= 0.5]
        print("  decode_overlap: mean %.2f  median %.2f" %
              (sum(ovs) / len(ovs), sorted(ovs)[len(ovs) // 2]))
        vk = sum(1 for r in valid if r["label"] == "REFUSAL")
        vlo, vhi = wilson(vk, len(valid))
        print("  VALID SUBSET (overlap>=0.5): n=%d  refusal %.2f  [CI %.2f-%.2f]"
              % (len(valid), vk / len(valid) if valid else 0, vlo, vhi))
    recs = [float(r["token_recovery"]) for r in scored if r.get("token_recovery")]
    if recs:
        validr = [r for r in scored if r.get("token_recovery")
                  and float(r["token_recovery"]) >= 0.5]
        print("  token_recovery: mean %.2f  median %.2f  items>=0.5: %d/%d" %
              (sum(recs) / len(recs), sorted(recs)[len(recs) // 2],
               len(validr), len(recs)))
        rk = sum(1 for r in validr if r["label"] == "REFUSAL")
        rlo, rhi = wilson(rk, len(validr))
        print("  VALID SUBSET (token_recovery>=0.5): n=%d  refusal %.2f  [CI %.2f-%.2f]"
              % (len(validr), rk / len(validr) if validr else 0, rlo, rhi))
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
