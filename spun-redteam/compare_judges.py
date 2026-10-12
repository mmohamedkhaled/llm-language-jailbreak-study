#!/usr/bin/env python3
"""Three-way labeler comparison: keyword heuristic vs local Mistral judge vs
frontier judge. Joins <base>_judged.csv and <base>_judged_frontier.csv on idx,
prints refusal counts/rates per labeler, pairwise agreement + Cohen's kappa,
and the disagreement rows for human review.

Usage: python3 compare_judges.py <base>_judged.csv <base>_judged_frontier.csv [...]
"""

import csv
import sys

from judge import binarize, cohen_kappa


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["idx"]: r for r in csv.DictReader(f)}


def compare(path_a, path_b):
    a, b = load(path_a), load(path_b)
    idxs = sorted(set(a) & set(b), key=lambda x: int(x))
    rows = []
    for i in idxs:
        kw = binarize(a[i]["label"], "REFUSAL")
        mj = binarize(a[i].get("judge_label", ""), "REFUSAL")
        gj = binarize(b[i].get("judge_label", ""), "REFUSAL")
        if kw is None:
            continue
        rows.append((i, kw, mj, gj))
    n = len(rows)
    print("== %s vs %s (n=%d) ==" % (path_a, path_b, n))
    for name, idx in (("keyword", 1), ("mistral-j", 2), ("frontier-j", 3)):
        r = sum(1 for t in rows if t[idx] == "R")
        print("  %-9s refusal %2d/%d = %.2f" % (name, r, n, r / n if n else 0))
    for name, x, y in (("kw~mistral", 1, 2), ("kw~frontier", 1, 3),
                       ("mistral~frontier", 2, 3)):
        pairs = [(t[x], t[y]) for t in rows
                 if t[x] is not None and t[y] is not None]
        agr = sum(1 for p, q in pairs if p == q)
        kap = cohen_kappa([p for p, _ in pairs], [q for _, q in pairs])
        print("  %-11s agree %.2f  kappa %.2f" % (name, agr / len(pairs), kap))
    dis = [t for t in rows if len({t[1], t[2], t[3]}) > 1]
    print("  3-way disagreements: %d rows%s" % (
        len(dis), (" (" + ",".join(t[0] for t in dis[:15]) + ")")
        if dis else ""))
    both_c = [t[0] for t in rows if t[2] == "C" and t[3] == "C"]
    print("  both LLM judges say COMPLY: %d rows%s" % (
        len(both_c), (" (" + ",".join(both_c[:15]) + ")") if both_c else ""))
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        compare(args[i], args[i + 1])
