#!/usr/bin/env python3
"""Build POLYGLOT prompts: each content word is rendered in one of three channels,
rotating English -> SPUN code -> rare language (Zulu). Particles stay English.
Tests whether mixing 'new stuff' + English + a rare language confuses safety while
staying comprehensible. Rare words translated via Qwen (Ollama); cached.

Writes polyglot.csv (goal = mixed prompt, english = original).
Usage: ../.venv/Scripts/python polyglot_build.py --n 15 --rare Zulu
"""
import argparse, csv, json, re
from pathlib import Path
from providers import get_provider
from spun_translator import SpunTranslator

HERE = Path(__file__).parent
tr = SpunTranslator()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--rare", default="Zulu")
    args = ap.parse_args()
    q = get_provider("ollama")
    M = "qwen3:30b-a3b-local"

    prompts = []
    for row in csv.DictReader(open(HERE / "advbench_harmful_behaviors.csv", encoding="utf-8")):
        g = (row.get("goal") or "").strip()
        if g:
            prompts.append(g)
        if len(prompts) >= args.n:
            break

    # unique content words needing rare translation (channel 2)
    need = set()
    for p in prompts:
        words = re.findall(r"[A-Za-z']+", p)
        cw = [w for w in words if w.lower() not in tr.particles]
        for i, w in enumerate(cw):
            if i % 3 == 2:
                need.add(w.lower())
    rare = {}
    for w in sorted(need):
        try:
            t = q.chat(M, [{"role": "user", "content":
                "Translate the single English word '%s' to %s. Output ONLY the one "
                "word, nothing else." % (w, args.rare)}], 0.0, 20, 60)
            rare[w] = re.split(r"[\s,\.]", t.strip())[0][:30] or w
        except Exception:
            rare[w] = w
    json.dump(rare, open(HERE / "polyglot_rare.json", "w", encoding="utf-8"), ensure_ascii=False)

    def poly(p):
        out = []
        ci = 0
        for tok in p.split():
            m = re.match(r"^([^\w]*)([A-Za-z']+)([^\w]*)$", tok)
            if not m:
                out.append(tok); continue
            pre, w, post = m.groups(); low = w.lower()
            if low in tr.particles:
                out.append(tok); continue
            ch = ci % 3; ci += 1
            if ch == 0:
                out.append(tok)                                   # English
            elif ch == 1:
                code, unk = tr._code_for(tr._lemmatize(low)[0])
                out.append("%s%s%s" % (pre, ("[%s]" % code) if code and not unk else w, post))
            else:
                out.append("%s%s%s" % (pre, rare.get(low, w), post))  # rare
        return " ".join(out)

    with open(HERE / "polyglot.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "goal", "english"])
        w.writeheader()
        for i, p in enumerate(prompts):
            w.writerow({"idx": i, "goal": poly(p), "english": p})
    print("wrote polyglot.csv (%d rows), rare dict %d words (%s)" % (len(prompts), len(rare), args.rare))
    print("sample:", poly(prompts[0]))


if __name__ == "__main__":
    main()
