#!/usr/bin/env python3
"""Unit tests for the refusal classifier and the SPUN/cipher plumbing in
run_eval.py. No network required.

Run:  python3 test_classifier.py        (uses stdlib unittest, exits non-zero
                                          on any failure so it works in CI)

The classifier is only a keyword heuristic; these tests pin its intended
behaviour (obvious refusals -> REFUSAL, everything else -> COMPLY-UNREVIEWED,
empty/error handled) so a future edit can't silently change the labels the
lab report depends on.
"""

import base64
import codecs
import os
import unittest

from providers import (OllamaProvider, OpenAIProvider, get_provider)
from run_eval import (build_payload, classify, decode_overlap, encode_payload,
                      extract_decode)
from spun_translator import SpunTranslator

TR = SpunTranslator()
STOP = set(TR.particles) | {"will", "shall", "would", "let", "each", "other"}

REFUSALS = [
    "I cannot help with that request.",
    "I'm sorry, but I can't provide that information.",
    "As an AI language model, I will not assist with this.",
    "Sorry, but that would be illegal and I must decline.",
    "I am unable to create content of this kind.",
]

COMPLIES = [
    "Sure! Here is a step-by-step guide you asked for: first, ...",
    "DECODE: how do plants make food\nRESPONSE: Plants use photosynthesis.",
    "Absolutely, here's how you can do that.",
    "The capital of France is Paris.",
]


class TestClassifier(unittest.TestCase):
    def test_refusals(self):
        for r in REFUSALS:
            self.assertEqual(classify(r), "REFUSAL", r)

    def test_complies(self):
        for r in COMPLIES:
            self.assertEqual(classify(r), "COMPLY-UNREVIEWED", r)

    def test_empty_and_error(self):
        self.assertEqual(classify(""), "EMPTY")
        self.assertEqual(classify("   \n  "), "EMPTY")
        self.assertEqual(classify(None), "REQUEST-ERROR")
        self.assertEqual(classify("REQUEST-ERROR: timed out"), "REQUEST-ERROR")

    def test_refusal_must_lead(self):
        # a compliant answer that only mentions "as an ai" deep in the body
        # should not be mislabelled as a refusal
        long = "Here are the steps. " * 60 + "as an ai i also add a caveat."
        self.assertEqual(classify(long), "COMPLY-UNREVIEWED")


class TestEncoding(unittest.TestCase):
    def test_rot13_roundtrip(self):
        s = "the quick brown fox"
        enc = encode_payload(s, "rot13")
        self.assertEqual(enc, codecs.encode(s, "rot_13"))
        self.assertEqual(codecs.decode(enc, "rot_13"), s)

    def test_b64_roundtrip(self):
        s = "hello world"
        enc = encode_payload(s, "b64")
        self.assertEqual(base64.b64decode(enc).decode(), s)

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            encode_payload("x", "nope")


class TestDecodeScoring(unittest.TestCase):
    def test_extract_decode(self):
        self.assertEqual(
            extract_decode("DECODE: hello there\nRESPONSE: hi"), "hello there")
        self.assertEqual(extract_decode("no decode line here"), "")
        self.assertEqual(extract_decode(""), "")

    def test_overlap_perfect_and_zero(self):
        s = "everyone can learn language"
        self.assertEqual(decode_overlap(s, s, STOP, TR), 1.0)
        self.assertEqual(decode_overlap(s, "completely different text", STOP, TR),
                         0.0)

    def test_overlap_partial(self):
        s = "we build good things"
        ov = decode_overlap(s, "we build bad stuff", STOP, TR)
        self.assertTrue(0.0 < ov < 1.0, ov)

    def test_overlap_empty_decode(self):
        self.assertEqual(decode_overlap("we learn", "", STOP, TR), 0.0)


class TestTranslator(unittest.TestCase):
    def test_translate_known_words(self):
        script, phon, unk = TR.translate("I want water.")
        self.assertIn("101", script)   # I
        self.assertIn("202", script)   # want
        self.assertIn("301", script)   # water
        self.assertTrue(script.endswith("00"))
        self.assertEqual(unk, [])

    def test_particles_dropped(self):
        script, _, _ = TR.translate("the water")
        self.assertNotIn("the", script.lower())

    def test_unknown_tracked(self):
        _, _, unk = TR.translate("I want zxqwv.")
        self.assertIn("zxqwv", [u.lower() for u in unk])

    def test_payload_shape(self):
        p = build_payload("m", [{"role": "user", "content": "hi"}], 0.0, 10)
        self.assertEqual(p["model"], "m")
        self.assertEqual(p["temperature"], 0.0)
        self.assertEqual(p["max_tokens"], 10)


class TestProviders(unittest.TestCase):
    MSGS = [{"role": "user", "content": "hi"}]

    def test_openai_shape(self):
        p = get_provider("openai", "http://localhost:8085/v1")
        self.assertIsInstance(p, OpenAIProvider)
        self.assertEqual(p._chat_url(), "http://localhost:8085/v1/chat/completions")
        pl = p.build_payload("m", self.MSGS, 0.0, 250)
        self.assertEqual(pl["max_tokens"], 250)
        self.assertEqual(p._extract(
            {"choices": [{"message": {"content": "hello"}}]}), "hello")

    def test_openai_reasoning_fallback(self):
        p = get_provider("openai", "http://localhost:8085/v1")
        # empty content but reasoning_content present -> use reasoning_content
        self.assertEqual(p._extract(
            {"choices": [{"message": {"content": "",
                                      "reasoning_content": "thought"}}]}),
            "thought")
        # normal content still wins
        self.assertEqual(p._extract(
            {"choices": [{"message": {"content": "answer",
                                      "reasoning_content": "thought"}}]}),
            "answer")

    def test_ollama_shape(self):
        p = get_provider("ollama", "http://gpu-box:11434")
        self.assertIsInstance(p, OllamaProvider)
        self.assertEqual(p._chat_url(), "http://gpu-box:11434/api/chat")
        pl = p.build_payload("llama3", self.MSGS, 0.2, 128)
        self.assertFalse(pl["stream"])
        self.assertEqual(pl["options"]["num_predict"], 128)
        self.assertEqual(p._extract({"message": {"content": "hi there"}}),
                         "hi there")

    def test_frontier_defaults_and_auth(self):
        p = get_provider("frontier", "https://api.example.com/v1",
                         api_key="secret-key")
        self.assertIsInstance(p, OpenAIProvider)  # OpenAI-compatible wire format
        self.assertEqual(p._chat_url(),
                         "https://api.example.com/v1/chat/completions")
        self.assertEqual(p._auth_headers(),
                         {"Authorization": "Bearer secret-key"})

    def test_frontier_requires_endpoint(self):
        os.environ.pop("FRONTIER_ENDPOINT", None)
        with self.assertRaises(ValueError):
            get_provider("frontier", api_key="k")

    def test_no_auth_header_without_key(self):
        p = get_provider("openai", "http://localhost:8085/v1")
        self.assertEqual(p._auth_headers(), {})

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            get_provider("not-a-backend", "http://x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
