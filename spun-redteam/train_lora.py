#!/usr/bin/env python3
"""Minimal LoRA fine-tune of Mistral-7B for the SPUN native-fluency experiment.

Memory-fitting for a 16 GB GPU: base weights frozen in bf16, LoRA on the
attention projections only, gradient checkpointing, batch 1 + grad-accum, short
sequences. No bitsandbytes (unreliable on Windows) — LoRA freezes the base so no
optimizer state is stored for it, which is what keeps this in budget.

Usage:
  ../.venv/Scripts/python train_lora.py --data train_translate.jsonl \
      --out lora_translate --epochs 2 [--max-steps 2 for a fit test]
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

MODEL_DIR = Path(__file__).parent.parent / "Mistral-7B-Instruct-v0.3"


class JsonlSFT(Dataset):
    def __init__(self, path, tok, max_len=256):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        user = (r["instruction"] + "\n" + r["input"]).strip()
        msgs = [{"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        full = prompt + r["output"] + self.tok.eos_token
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = self.tok(full, add_special_tokens=False)["input_ids"][:self.max_len]
        labels = list(f_ids)
        for j in range(min(len(p_ids), len(labels))):   # mask the prompt tokens
            labels[j] = -100
        return {"input_ids": f_ids, "labels": labels}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        n = m - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * n)
        lab.append(b["labels"] + [-100] * n)
        att.append([1] * len(b["input_ids"]) + [0] * n)
    return (torch.tensor(ids), torch.tensor(lab), torch.tensor(att))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 for a fit test")
    ap.add_argument("--resume", default=None,
                    help="existing adapter dir to continue training from")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.resume:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        print("resumed adapter from", args.resume)
    else:
        lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = JsonlSFT(args.data, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    steps = (len(dl) // args.accum) * args.epochs
    if args.max_steps:
        steps = args.max_steps
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    sch = get_cosine_schedule_with_warmup(opt, int(0.03 * steps) + 1, steps)

    model.train()
    step, done = 0, False
    for ep in range(args.epochs):
        for bi, (ids, lab, att) in enumerate(dl):
            ids, lab, att = ids.to(model.device), lab.to(model.device), att.to(model.device)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            (out.loss / args.accum).backward()
            if (bi + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
                step += 1
                if step % 5 == 0 or args.max_steps:
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    print("step %d/%d loss %.3f peakGPU %.1fGB"
                          % (step, steps, out.loss.item(), mem), flush=True)
                if args.max_steps and step >= args.max_steps:
                    done = True
                    break
        if done:
            break

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved adapter to %s" % args.out)


if __name__ == "__main__":
    main()
