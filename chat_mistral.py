"""Quick interactive chat with the local Mistral-7B-Instruct-v0.3.

Run:  .venv\\Scripts\\python chat_mistral.py            (interactive)
      .venv\\Scripts\\python chat_mistral.py "question"  (one-shot)
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "Mistral-7B-Instruct-v0.3"

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "14GiB", "cpu": "48GiB"},
)
model.eval()
print("[loaded in %.1fs on %s]" % (time.time() - t0, model.device), flush=True)


def ask(history):
    inputs = tok.apply_chat_template(history, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    n = inputs["input_ids"].shape[1]
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()


if len(sys.argv) > 1:
    print(ask([{"role": "user", "content": " ".join(sys.argv[1:])}]))
    sys.exit()

history = []
while True:
    try:
        q = input("\nyou> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q in ("exit", "quit"):
        break
    history.append({"role": "user", "content": q})
    a = ask(history)
    history.append({"role": "assistant", "content": a})
    print("mistral> " + a)
