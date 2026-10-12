#!/usr/bin/env python3
"""Expand the SPUN lexicon so it can actually EXPRESS AdvBench prompts.

We proved the jailbreak blocker is coverage: only 40% of harmful content words
have a code. This assigns a fresh numeric code to every AdvBench content word
that is currently uncovered, and writes the enlarged lexicon.

This is purely a DICTIONARY (number<->word). No harmful behaviour is authored or
trained; the codes just let the harmful *request* survive encoding so we can test
whether the model, once it can decode it, refuses (concept-level safety) or not.

Backup: the original 100-word lexicon remains in spun_lexicon_v3.json.
Output:  spun_lexicon.json (expanded, used by all scripts)
Usage:   ../.venv/Scripts/python build_lexicon_expand.py
"""

import json
import re
from pathlib import Path

from spun_translator import SpunTranslator

HERE = Path(__file__).parent
tr = SpunTranslator()

# collect uncovered content words across the whole AdvBench file
import csv
words = {}
for row in csv.DictReader(open(HERE / "advbench_harmful_behaviors.csv", encoding="utf-8")):
    g = (row.get("goal") or "").strip()
    for w in re.findall(r"[a-z']+", g.lower()):
        if w in tr.particles:
            continue
        base, _ = tr._lemmatize(w)
        code, is_unknown = tr._code_for(base)
        if is_unknown or code is None:
            words[base] = words.get(base, 0) + 1

# assign new codes starting at 1000, most frequent first (stable, deterministic)
new = sorted(words.items(), key=lambda kv: (-kv[1], kv[0]))
cfg = dict(tr.cfg)
lex = dict(cfg["lexicon"])
code = 1000
added = []
for w, _ in new:
    while str(code) in lex:
        code += 1
    lex[str(code)] = [w]
    added.append((str(code), w))
    code += 1
cfg["lexicon"] = lex
cfg["version"] = "2.0-expanded"
# put every new code in the 'nouns' bucket (best-effort category for probing)
cfg.setdefault("categories", {}).setdefault("nouns", [])
cfg["categories"]["nouns"] += [c for c, _ in added]

with open(HERE / "spun_lexicon.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print("added %d new codes (now %d total)" % (len(added), len(lex)))
print("sample new entries:", added[:10])

# report new coverage on AdvBench
tr2 = SpunTranslator(HERE / "spun_lexicon.json")
def cov(text):
    ws = [w for w in re.findall(r"[a-z']+", text.lower()) if w not in tr2.particles]
    if not ws:
        return None
    _, _, unk = tr2.translate(text)
    return 1 - len(unk) / len(ws)
harm = []
for row in csv.DictReader(open(HERE / "advbench_harmful_behaviors.csv", encoding="utf-8")):
    g = (row.get("goal") or "").strip()
    if g:
        harm.append(g)
    if len(harm) >= 50:
        break
c = [cov(t) for t in harm if cov(t) is not None]
print("AdvBench coverage now: %.0f%% (was 40%%)" % (100 * sum(c) / len(c)))
