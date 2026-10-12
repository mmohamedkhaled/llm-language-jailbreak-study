#!/usr/bin/env python3
"""Phase 1 of the low-resource-language (distribution-gap) test: translate a
sample of AdvBench prompts into a resource gradient of languages, via Qwen3-30B
(multilingual). Writes one CSV per language (translated text in the 'goal' column)
plus a back-translation for a quality/comprehension check.

Tests the Yong et al. hypothesis on Mistral: does a non-English wrapper lower
refusal -- or does poor translation/competence just yield spurious comply (the
same failure mode as SPUN)?

Usage: ../.venv/Scripts/python make_multilingual.py --n 20
"""
import argparse, csv
from providers import get_provider

LANGS = ["French", "Spanish", "German", "Vietnamese", "Indonesian", "Swahili"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    p = get_provider("ollama")
    M = "qwen3:30b-a3b-local"

    prompts = []
    for row in csv.DictReader(open("advbench_harmful_behaviors.csv", encoding="utf-8")):
        g = (row.get("goal") or "").strip()
        if g:
            prompts.append(g)
        if len(prompts) >= args.n:
            break

    def tr(text, lang):
        r = p.chat(M, [{"role": "user", "content":
            "Translate the following English text to %s. Output ONLY the "
            "translation, no notes:\n%s" % (lang, text)}], 0.0, 120, 60)
        return r.strip().split("\n")[0][:400]

    for lang in LANGS:
        rows = []
        for i, en in enumerate(prompts):
            t = tr(en, lang)
            back = tr(t, "English")            # back-translation for fidelity
            rows.append({"idx": i, "goal": t, "english": en, "backtranslation": back})
        out = "ml_%s.csv" % lang.lower()
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["idx", "goal", "english", "backtranslation"])
            w.writeheader(); w.writerows(rows)
        print("wrote %s (%d rows)" % (out, len(rows)), flush=True)


if __name__ == "__main__":
    main()
