#!/usr/bin/env python3
"""Localize cross-lingual refusal: sweep ALL layers in one model load and report,
per layer, how strongly harmful prompts in each representation project onto the
English-derived refusal direction. Shows WHERE language-invariant safety lives.

No generation, no harmful output -- pure activation geometry.
Usage:  ../.venv/Scripts/python refusal_probe_sweep.py --n 40
"""
import argparse, csv
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from run_eval import encode_payload
from spun_translator import SpunTranslator
from refusal_probe import HARMLESS

MD = Path(__file__).parent.parent / "Mistral-7B-Instruct-v0.3"


@torch.no_grad()
def all_layer_means(model, tok, prompts, batch=8):
    """Return tensor [num_layers, D]: mean last-token vector per layer."""
    acc = None
    count = 0
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                 tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple[L+1]
        lengths = enc["attention_mask"].sum(1) - 1
        layers = torch.stack(hs)                       # [L+1, B, T, D]
        for b in range(layers.shape[1]):
            v = layers[:, b, lengths[b], :].float()    # [L+1, D]
            acc = v if acc is None else acc + v
            count += 1
    return acc / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    tr = SpunTranslator()
    tok = AutoTokenizer.from_pretrained(MD)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16,
                                                 device_map={"": 0}).eval()

    harmful = []
    for row in csv.DictReader(open(Path(__file__).parent /
                              "advbench_harmful_behaviors.csv", encoding="utf-8")):
        g = (row.get("goal") or "").strip()
        if g: harmful.append(g)
        if len(harmful) >= args.n: break
    harmless = HARMLESS[:args.n]

    m_harm = all_layer_means(model, tok, harmful)
    m_safe = all_layer_means(model, tok, harmless)
    d = m_harm - m_safe                                   # [L+1, D] per-layer dir
    d_hat = d / d.norm(dim=1, keepdim=True)

    spun = [tr.translate(t)[0] for t in harmful]
    rot13 = [encode_payload(t, "rot13") for t in harmful]
    a_en = m_harm
    a_sp = all_layer_means(model, tok, spun)
    a_ro = all_layer_means(model, tok, rot13)
    base = (m_safe * d_hat).sum(1)                        # harmless english baseline

    print("layer |  en  | spun% | rot13%   (proj onto per-layer refusal dir, "
          "rel. to harmless-en)")
    L = a_en.shape[0]
    for l in range(0, L, 2):
        en = ((a_en[l] * d_hat[l]).sum() - base[l]).item()
        sp = ((a_sp[l] * d_hat[l]).sum() - base[l]).item()
        ro = ((a_ro[l] * d_hat[l]).sum() - base[l]).item()
        if abs(en) < 1e-6:
            continue
        print("  %2d  | %5.2f | %4.0f%% | %4.0f%%" %
              (l, en, 100 * sp / en, 100 * ro / en))


if __name__ == "__main__":
    main()
