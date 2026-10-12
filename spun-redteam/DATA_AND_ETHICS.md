# Data availability & responsible-release note

This is a **defensive red-team measurement** study. It quantifies whether an
encoding channel (an invented numeric conlang **SPUN**, classic ciphers, a
hybrid code-switch, or a conlang-fine-tuned model) changes an instruction-tuned
model's refusal rate on the public
[AdvBench](https://github.com/llm-attacks/llm-attacks) harmful-behaviour set,
and compares that "language" family against social (multi-turn) and
turn-structure (prefill) attacks.

## What this repository DOES contain

- The full evaluation **harness and all source code** (`*.py`).
- The **SPUN translator, lexicons, and grammar** (`spun_*.json`, `spun_translator.py`).
- **Benign** SPUN training/eval data (`train_*.jsonl`, `test_fluency.jsonl`) — these
  are harmless translation pairs (e.g. *"I see a good knowledge."*), used for the
  benign fluency fine-tuning arms.
- The **public AdvBench prompt set** (`advbench_harmful_behaviors.csv`) — prompts
  only. This file is already public and is the standard benchmark; it is included
  so the harness runs out of the box.
- The **write-ups and rendered reports** (`docs/PROJECT_WRITEUP.md`, `docs/FINDINGS.md`,
  `docs/LAB_REPORT.md`, `docs/findings_page.html`, `docs/summary_page.html`), which report
  aggregate statistics (refusal/compliance rates, Wilson CIs, decode fidelity).
- LoRA adapter **metadata only** (`lora_*/adapter_config.json`, `README.md`).

## What this repository deliberately does NOT contain

To avoid distributing potentially harmful material, the following were withheld
(they are produced locally by the harness and are covered by `.gitignore`):

1. **Elicited model completions.** Every `*_50.csv`, `*_judged*.csv`, `adv_*.csv`,
   `frontier*.csv`, `mm_*.csv`, etc. holds verbatim model *responses* to AdvBench
   prompts. A minority of those responses are actionable harmful content. They are
   treated as measurement data — adjudicated, counted, and reported in aggregate —
   but **not published**.
2. **Human-adjudication workbooks** (`adjudication/`) and dump files (`adj_*.txt`),
   for the same reason.
3. **Secrets.** No API keys. Key files (`*.key`, `.env`) are git-ignored; supply your own key via
   the `FRONTIER_API_KEY` environment variable.
4. **Large regenerable binaries.** The local model weights
   (`Mistral-7B-Instruct-v0.3/`), the Python venv, and the LoRA `*.safetensors`
   adapter weights. Download the model from its original source; retrain the
   adapters with `train_lora.py` (benign data is included).

## Reproducing the results

1. `pip install -r requirements.txt`
2. Obtain `Mistral-7B-Instruct-v0.3` (or point `--provider`/`--endpoint` at any
   local or remote OpenAI-compatible / Ollama / frontier-API server).
3. Run the arms as shown in [README.md](README.md); score with
   `summarize_results.py`, add the second labeler with `judge.py`, and build
   review workbooks with `adjudicate.py`.

The aggregate numbers you should reproduce are tabulated in
[PROJECT_WRITEUP.md](docs/PROJECT_WRITEUP.md) §0.

## Headline finding

No language channel — invented, real, mixed, or fine-tuned — reliably jailbreaks a
well-aligned model; safety is **concept-level**, not surface-level. The only
genuine jailbreaks found were a social (multi-turn) attack on a weak model and a
prefill (turn-structure) attack on one frontier model — **neither is a language**.
See the write-up for the full argument and the measurement caveats (keyword
detection massively overstates attack success; a language-agnostic judge + decode
fidelity gate is required).

## Attack prompts that ARE in the code

The harness necessarily includes the *wrappers* used to test refusal behaviour:
the multi-turn fiction-framing prompts and the ROT13 + persona stack in
`eval_advanced.py`, and the forced-prefix ("Sure, here are...") prefill in
`prefill_eval.py` / `prefill_provider.py`. These are generic, widely published
techniques (role-play framing, prefill, cipher wrapping); they contain no
harmful content of their own and are included so the results can be audited
and the defences evaluated. The study's own finding is that the multi-turn
framing is the only wrapper that worked on the weak local model; do not run
these against systems you do not own or have permission to test, and follow
each provider's usage policy and disclosure process.

## Model naming

Hosted models under test are anonymised in the write-ups (e.g. "Frontier-A")
and the hosted provider is the generic `frontier` adapter, which has no
built-in endpoint: supply your own with `--endpoint` or `FRONTIER_ENDPOINT`.

## Attribution

Prompts come from AdvBench (MIT-licensed):
Zou, Wang, Carlini, Nasr, Kolter, Fredrikson. *Universal and Transferable
Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043, 2023.
Local model: [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) (Apache-2.0).
