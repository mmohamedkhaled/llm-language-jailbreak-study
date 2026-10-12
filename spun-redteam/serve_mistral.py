#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server for a local HF model (stdlib HTTP).

Serves POST /v1/chat/completions so run_eval.py can target it with
--endpoint http://localhost:<port>/v1

Run with a Python that has torch+transformers, e.g.:
  /path/to/venv/bin/python serve_mistral.py --model ./Mistral-7B-Instruct-v0.3
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCK = threading.Lock()
MODEL = None
TOKENIZER = None


def load(model_path, dtype):
    global MODEL, TOKENIZER
    t0 = time.time()
    TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=getattr(torch, dtype),
        # "auto" spills layers to CPU RAM if the GPU (16 GB RTX 5080) runs short
        device_map="auto",
        max_memory={0: "14GiB", "cpu": "48GiB"},
    )
    MODEL.eval()
    print("[serve] model loaded in %.1fs" % (time.time() - t0), flush=True)


def chat(messages, temperature, max_tokens):
    # Mistral v0.3 template has no system role: merge system into first user turn
    msgs = list(messages)
    if msgs and msgs[0]["role"] == "system":
        sys_txt = msgs.pop(0)["content"]
        if msgs:
            msgs[0]["content"] = sys_txt + "\n\n" + msgs[0]["content"]
        else:
            msgs = [{"role": "user", "content": sys_txt}]

    prompt = TOKENIZER.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        out = MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
            messages = body["messages"]
            temperature = float(body.get("temperature", 0.0))
            max_tokens = int(body.get("max_tokens", 256))
        except (ValueError, KeyError) as e:
            self.send_error(400, str(e))
            return
        t0 = time.time()
        try:
            with LOCK:
                text = chat(messages, temperature, max_tokens)
        except Exception as e:
            self.send_error(500, str(e))
            return
        resp = {
            "id": "local-%d" % int(time.time() * 1000),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "local"),
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text},
                 "finish_reason": "stop"}
            ],
            "usage": {},
        }
        payload = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        print("[chat] %.1fs tokens_out_len=%d" % (time.time() - t0, len(text.split())),
              flush=True)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            payload = json.dumps({"data": [{"id": "local"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(200)
            self.end_headers()

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../Mistral-7B-Instruct-v0.3")
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (no auth: use 0.0.0.0 only on a trusted LAN)")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    load(args.model, args.dtype)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("[serve] listening on %s:%d" % (args.host, args.port), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
