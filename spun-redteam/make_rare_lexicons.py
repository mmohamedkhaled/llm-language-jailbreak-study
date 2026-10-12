#!/usr/bin/env python3
"""Build EN->rare-language word dictionaries for the --ml-hybrid arm.

Uses the frontier API to translate the targeted word list (same set as the
targeted-SPUN arm, read from spun_lexicon_targeted.json) into German
(high-resource control), Icelandic and Welsh (rare). Words are translated in
small chunks, format 'english=translation' one per line; results accumulate
across retries. A back-translation sample verifies quality. Writes
ml_word_lexicons.json: {"german": {en: word}, ...}

Usage: ../.venv/Scripts/python make_rare_lexicons.py [--sample N] [--model M]
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from providers import get_provider

OUT = Path(__file__).parent / "ml_word_lexicons.json"
LANGS = ["German", "Icelandic", "Welsh"]
CHUNK = 12


def ask(provider, model, system, user, max_tokens=4000):
    return provider.chat(
        model,
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        temperature=0.0, max_tokens=max_tokens).strip()


def parse_pairs(text, wanted):
    """Parse 'english=translation' lines; keep those whose left side
    (leniently normalized) matches a wanted word."""
    out = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        en, tr = line.split("=", 1)
        en = en.strip().strip("`*").lower()
        tr = tr.strip().strip("`*")
        if not en or not tr:
            continue
        norm = re.sub(r"[^\w\s-]", "", en, flags=re.UNICODE).strip().lower()
        norm = norm[3:] if norm.startswith("to ") else norm
        for w in wanted:
            if norm == w or norm.startswith(w + " ") or norm == w + " word":
                out[w] = tr
                break
    return out


def translate_lang(provider, model, words, lang):
    sys_p = ("This is a bilingual lexicography task for a linguistics "
             "database. For each English word on the list, give its standard "
             "%s dictionary form (nouns singular, verbs infinitive). Output "
             "format: exactly one line per word, 'english=%s'. No commentary, "
             "no header. Every input word must appear exactly once on the "
             "left side." % (lang, lang))
    done = {}
    chunks = [words[i:i + CHUNK] for i in range(0, len(words), CHUNK)]
    for ci, chunk in enumerate(chunks):
        for attempt in range(3):
            try:
                got = parse_pairs(
                    ask(provider, model, sys_p, "\n".join(chunk)), chunk)
            except Exception as e:  # noqa: BLE001 - retry transport errors
                print("  chunk %d attempt %d: %s" % (ci, attempt + 1, e))
                time.sleep(3)
                continue
            done.update(got)
            missing = [w for w in chunk if w not in done]
            if not missing:
                break
            print("  chunk %d attempt %d: %d/%d, retrying %s"
                  % (ci, attempt + 1, len(got), len(chunk),
                     ",".join(missing)))
    return done


def normalize_en(word):
    w = word.lower().strip()
    if w.startswith("to "):
        w = w[3:]
    return w.split(";")[0].split(",")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=15,
                    help="back-translation sample size per language")
    ap.add_argument("--model", required=True, help="model ID from your provider console")
    ap.add_argument("--verify", action="store_true",
                    help="re-check an existing ml_word_lexicons.json "
                         "(full back-translation, no rewriting)")
    args = ap.parse_args()
    provider = get_provider("frontier", None, None, 300)  # FRONTIER_API_KEY env
    if args.verify:
        data = json.loads(OUT.read_text(encoding="utf-8"))
        for lang, pairs in sorted(data.items()):
            check_back(provider, args.model, lang.capitalize(),
                       sorted(pairs.items()))
        return 0
    words = sorted(json.loads(
        Path("spun_lexicon_targeted.json").read_text(encoding="utf-8")))
    print("%d source words, translator=%s" % (len(words), args.model))
    result = {}
    for lang in LANGS:
        pairs = translate_lang(provider, args.model, words, lang)
        print("%s: %d/%d translated" % (lang, len(pairs), len(words)))
        sample = random.sample(sorted(pairs.items()),
                               min(args.sample, len(pairs)))
        check_back(provider, args.model, lang, sample)
        result[lang.lower()] = pairs
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False,
                              sort_keys=True), encoding="utf-8")
    print("wrote %s" % OUT.name)
    return 0


def check_back(provider, model, lang, pairs_list):
    """Back-translate (en, translation) pairs and report mismatches."""
    sys_b = ("Translate single %s words to English for a linguistics "
             "database. Output format: one line per word, "
             "'%s=english'. No commentary." % (lang, lang))
    try:
        back = parse_pairs(
            ask(provider, model, sys_b, "\n".join(w for _, w in pairs_list)),
            [w for _, w in pairs_list])
    except Exception as e:  # noqa: BLE001
        back = {}
        print("  back-translation call failed: %s" % e)
    mism = [(en, tr, back.get(tr, "?"))
            for en, tr in pairs_list
            if normalize_en(back.get(tr, "")) != en]
    print("  back-translation check: %d/%d ok"
          % (len(pairs_list) - len(mism), len(pairs_list)))
    for en, tr, got in mism:
        print("   MISS %s -> %s -> %s"
              % (en, tr.encode("ascii", "backslashreplace").decode(), got))


if __name__ == "__main__":
    sys.exit(main())
