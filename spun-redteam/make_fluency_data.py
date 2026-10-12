#!/usr/bin/env python3
"""Build a clean, GENERALISABLE benign SPUN-fluency dataset.

Fixes from the first attempt (which memorised junk -> 0 generalisation):
  * only VERIFIABLE data: word<->code drills and exact sentence<->script pairs.
    NO model-generated ("self-distilled") responses -- the base model can't read
    SPUN, so those were boilerplate unrelated to meaning.
  * grammatical, DIVERSE sentences (varied subjects/verbs/objects/tense/number).
  * a held-out TEST split (test_fluency.jsonl) of unseen sentences, so we can
    tell real compositional fluency from memorisation.

All benign; built only from the lexicon's own vocabulary.

Usage:  ../.venv/Scripts/python make_fluency_data.py --sents 800 --holdout 120
"""

import argparse
import json
import random
from pathlib import Path

from spun_translator import SpunTranslator

random.seed(7)
tr = SpunTranslator()
HERE = Path(__file__).parent

# grammatical building blocks, all in-lexicon
SUBJ = ["I", "We", "You", "They"]           # base-form verb agreement
VERB = ["want", "learn", "help", "create", "see", "share", "eat", "do"]
NOUN = ["water", "food", "work", "language", "knowledge", "peace", "hope",
        "freedom", "truth", "life", "the world", "a good idea", "a new tool",
        "a big place", "an old thing"]
ADV = ["together", "always", "never", "fast"]


def grammatical_sentences(n):
    """Yield diverse, grammatical, cleanly-encodable benign sentences."""
    out = set()
    tries = 0
    while len(out) < n and tries < n * 60:
        tries += 1
        s_, v, o = random.choice(SUBJ), random.choice(VERB), random.choice(NOUN)
        form = random.randint(0, 8)
        if form == 0:
            s = "%s %s %s." % (s_, v, o)
        elif form == 1:
            s = "%s want to %s %s." % (s_, v, o)
        elif form == 2:
            s = "%s can %s %s." % (s_, v, o)
        elif form == 3:
            s = "%s %s %s %s." % (s_, v, o, random.choice(ADV))
        elif form == 4:
            s = "%s must %s %s together." % (s_, v, o)
        elif form == 5:
            s = "%s will %s %s." % (s_, v, o)          # future
        elif form == 6:
            s = "%s do not want %s." % (s_, o)         # negation
        elif form == 7:
            s = "Can %s %s %s?" % (s_.lower(), v, o)    # question
        else:
            s = "%s %s %s because knowledge is good." % (s_, v, o)
        _, _, unk = tr.translate(s)
        if not unk:                       # only keep fully-encodable sentences
            out.add(s)
    return list(out)


def drills():
    rows = []
    for code, words in tr.lexicon.items():
        w = words[0]
        rows.append({"instruction": "What is the SPUN numeric code for the "
                     "English word '%s'?" % w, "input": "", "output": code})
        rows.append({"instruction": "What English word does the SPUN code %s "
                     "mean?" % code, "input": "", "output": w})
    for name, a in tr.affixes.items():
        rows.append({"instruction": "In SPUN, what does the affix '%s' mean?"
                     % a["code"], "input": "", "output": name.replace("_", " ")})
    return rows


def sent_rows(sents):
    rows = []
    for s in sents:
        script, _, _ = tr.translate(s)
        rows.append({"instruction": "Translate this SPUN numeric message to "
                     "English.", "input": "SPUN: " + script, "output": s})
        rows.append({"instruction": "Translate this English sentence to SPUN "
                     "numeric script.", "input": s, "output": script})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sents", type=int, default=800)
    ap.add_argument("--holdout", type=int, default=120)
    args = ap.parse_args()

    sents = grammatical_sentences(args.sents + args.holdout)
    random.shuffle(sents)
    test_sents = sents[:args.holdout]
    train_sents = sents[args.holdout:]

    train = drills() + sent_rows(train_sents)
    # duplicate drills once more so the compositional mapping isn't drowned out
    train += drills()
    random.shuffle(train)
    with open(HERE / "train_fluency.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(HERE / "test_fluency.jsonl", "w", encoding="utf-8") as f:
        for s in test_sents:                      # decode-only test rows
            script, _, _ = tr.translate(s)
            f.write(json.dumps({"input": "SPUN: " + script, "output": s}) + "\n")

    print("train_fluency.jsonl: %d rows (%d train sents x2 + drills x2)"
          % (len(train), len(train_sents)))
    print("test_fluency.jsonl:  %d held-out sentences (never seen in training)"
          % len(test_sents))
    print("sample train sentences:", train_sents[:4])


if __name__ == "__main__":
    main()
