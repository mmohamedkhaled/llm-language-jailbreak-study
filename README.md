# llm-language-jailbreak-study

Do encoding channels and invented languages (SPUN, ROT13, Base64, code-switching,
fine-tuned conlang) bypass LLM safety refusals? A defensive red-team measurement study.

- **[spun-redteam/](spun-redteam/README.md)** — the harness, SPUN language, benign
  training data and write-ups. Start with `spun-redteam/docs/PROJECT_WRITEUP.md`.
- **[chat_mistral.py](chat_mistral.py)** — small helper to chat with a local
  Mistral-7B-Instruct-v0.3 (weights are not included; see
  `spun-redteam/DATA_AND_ETHICS.md`).

Model completions, adjudication workbooks, weights and keys are deliberately not
published. Licensed MIT (`spun-redteam/LICENSE`).
