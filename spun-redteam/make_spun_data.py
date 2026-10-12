#!/usr/bin/env python3
"""Generate BENIGN SPUN training data for the native-fluency fine-tune.

Two datasets, matching the two experimental arms:
  train_translate.jsonl  SPUN script  -> English  (teaches decoding; the model
                         still routes through English concepts -> safety should
                         mostly survive)
  train_native.jsonl     SPUN script  -> a helpful English response, with NO
                         decode step (teaches the model to act on SPUN directly
                         -> the arm that tests whether safety is bypassed)

All training content is benign (templated from the lexicon's own benign
vocabulary + curated benign instructions). Harmful AdvBench is used ONLY at
evaluation, never in training. Responses for the native arm are self-distilled
from the local base model on the *English* instruction, so the fine-tune teaches
"understand SPUN and give the answer you'd have given in English".

Usage:
  ../.venv/Scripts/python make_spun_data.py --distill --limit-native 100
"""

import argparse
import json
import random
from pathlib import Path

from spun_translator import SpunTranslator

random.seed(0)
tr = SpunTranslator()

# benign vocabulary drawn from the lexicon (first gloss per code)
VERBS = ["do", "want", "eat", "go", "help", "create", "learn", "share", "see"]
NOUNS = ["thing", "water", "food", "place", "work", "tool", "language", "idea",
         "life", "world", "knowledge", "peace", "hope", "freedom", "truth"]
ADJS = ["good", "big", "new", "true", "easy"]
TEMPLATES = [
    "I want to {v} {n}.",
    "We {v} {n} together.",
    "Can you help me {v} {n}?",
    "Please {v} a {a} {n}.",
    "Everyone can {v} {n}.",
    "I {v} {n} every day.",
    "They want to {v} a {a} {n}.",
    "We must {v} {n} together.",
    "It is {a} to {v} {n}.",
    "I see a {a} {n}.",
]

EXTRA_BENIGN = [
    "The sun rises in the east.", "Children play in the park.",
    "I drink tea every morning.", "The teacher writes on the board.",
    "We walk home after school.", "The stars shine at night.",
    "Everyone can learn language.", "Life is hard but hope is strong.",
    "Together we build harmony.", "I want water.", "We learn together.",
    "I do not want conflict.", "I will create good things.",
    "Is knowledge freedom?", "We help each other.",
]


def gen_sentences(n):
    out = set(EXTRA_BENIGN)
    while len(out) < n:
        t = random.choice(TEMPLATES)
        s = t.format(v=random.choice(VERBS), n=random.choice(NOUNS),
                     a=random.choice(ADJS))
        script, _, unk = tr.translate(s)
        if not unk:                      # only keep cleanly-encodable sentences
            out.add(s)
    return list(out)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--limit-native", type=int, default=100)
    ap.add_argument("--distill", action="store_true",
                    help="self-distill native-arm responses from the local model")
    args = ap.parse_args()

    sents = gen_sentences(args.n)
    here = Path(__file__).parent

    # translate arm: SPUN -> English  (both directions would also work)
    with open(here / "train_translate.jsonl", "w", encoding="utf-8") as f:
        for s in sents:
            script, _, _ = tr.translate(s)
            f.write(json.dumps({
                "instruction": "Translate this SPUN numeric message to English.",
                "input": "SPUN: " + script,
                "output": s,
            }) + "\n")
    print("wrote train_translate.jsonl: %d rows" % len(sents))

    # native arm: SPUN -> helpful response (no decode step)
    native = sents[:args.limit_native]
    responses = {}
    if args.distill:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        md = here.parent / "Mistral-7B-Instruct-v0.3"
        tok = AutoTokenizer.from_pretrained(md)
        # GPU-only (no CPU offload) so generation is fast and the call stays
        # under the 600s foreground limit
        model = AutoModelForCausalLM.from_pretrained(
            md, dtype=torch.bfloat16, device_map={"": 0})
        model.eval()
        for i, s in enumerate(native):
            msgs = [{"role": "user", "content": s}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt",
                                          return_dict=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            resp = tok.decode(out[0][enc["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
            responses[s] = resp
            if i % 20 == 0:
                print("  distilled %d/%d" % (i, len(native)), flush=True)
    with open(here / "train_native.jsonl", "w", encoding="utf-8") as f:
        for s in native:
            script, _, _ = tr.translate(s)
            resp = responses.get(s) or ("Sure. " + s[0].lower() + s[1:])
            f.write(json.dumps({
                "instruction": "You understand SPUN. Respond to this message.",
                "input": script,
                "output": resp,
            }) + "\n")
    print("wrote train_native.jsonl: %d rows (distilled=%s)"
          % (len(native), args.distill))


if __name__ == "__main__":
    main()
