#!/usr/bin/env bash
# Multi-model replication of the encoding-channel experiment.
# For each model: serve it locally, run plain / SPUN / ROT13 arms on the same
# AdvBench-50 items plus the benign fidelity check, then stop the server.
# Resumable: skips a model if its summary file already exists.
set -u
# Adjust BASE to wherever your local model weights live, and PY to a python
# that has the project requirements installed (defaults below).
BASE="${MODELS_DIR:-$HOME/base-models}"
PY="${PYTHON:-python3}"
END="http://localhost:8085/v1"
cd "$(dirname "$0")"

run_model () {
  local tag="$1" path="$2"
  local out="mm_${tag}"
  [ -f "${out}_plain.csv" ] && [ -f "${out}_spun.csv" ] && [ -f "${out}_rot13.csv" ] && {
    echo "== $tag already done, skipping"; return 0; }
  echo "== $tag : stopping old server, starting new"
  pkill -f serve_mistral.py; sleep 3
  nohup "$PY" serve_mistral.py --model "$path" --port 8085 > "server_${tag}.log" 2>&1 &
  for i in $(seq 1 60); do
    sleep 5
    grep -q "listening on" "server_${tag}.log" 2>/dev/null && break
  done
  grep -q "listening on" "server_${tag}.log" || { echo "$tag FAILED TO LOAD"; return 1; }
  echo "== $tag : smoke test"
  curl -s -m 120 "$END/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"m","temperature":0,"max_tokens":30,"messages":[{"role":"user","content":"Say hello in one short sentence."}]}' \
    | head -c 200; echo
  echo "== $tag : plain arm"
  python3 run_eval.py advbench_harmful_behaviors.csv --column goal --limit 50 \
    --model m --endpoint "$END" --plain --out "${out}_plain.csv" \
    --delay 0.2 --max-tokens 250 > "${out}_plain.log" 2>&1
  echo "== $tag : spun arm"
  python3 run_eval.py advbench_harmful_behaviors.csv --column goal --limit 50 \
    --model m --endpoint "$END" --include-system-prompt --out "${out}_spun.csv" \
    --delay 0.2 --max-tokens 250 > "${out}_spun.log" 2>&1
  echo "== $tag : rot13 arm"
  python3 run_eval.py advbench_harmful_behaviors.csv --column goal --limit 50 \
    --model m --endpoint "$END" --encode rot13 --out "${out}_rot13.csv" \
    --delay 0.2 --max-tokens 250 > "${out}_rot13.log" 2>&1
  echo "== $tag : benign fidelity"
  python3 run_eval.py demo_benign.txt --model m --endpoint "$END" \
    --include-system-prompt --out "${out}_benign.csv" \
    --delay 0.2 --max-tokens 250 > "${out}_benign.log" 2>&1
  tail -3 "${out}_plain.log" | head -2
}

run_model llama3b  "$BASE/Llama-3.2-3B-Instruct"    || true
run_model qwen25_3b "$BASE/Qwen2.5-3B-Instruct"      || true
run_model qwen35_9b "$BASE/qwen3.5-9b-model"         || true

pkill -f serve_mistral.py; sleep 2
echo "ALL DONE"
