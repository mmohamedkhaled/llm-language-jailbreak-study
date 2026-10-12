#!/usr/bin/env python3
"""Offline unit tests for judge.py and adjudicate.py (no network).

Run: python3 test_judge.py
"""

import csv
import os
import tempfile
import unittest

from judge import (JUDGE_SYSTEM, binarize, cohen_kappa, judge_row,
                   parse_judge_reply)
from adjudicate import build, enrich, head_and_tail_flags


class TestParseJudgeReply(unittest.TestCase):
    def test_plain_words(self):
        self.assertEqual(parse_judge_reply("REFUSAL"), "REFUSAL")
        self.assertEqual(parse_judge_reply("COMPLY"), "COMPLY")
        self.assertEqual(parse_judge_reply("refusal\nbecause ..."), "REFUSAL")

    def test_leading_text(self):
        self.assertEqual(
            parse_judge_reply("1. REFUSAL\nThe assistant declines."),
            "REFUSAL")

    def test_none_and_garbage(self):
        self.assertEqual(parse_judge_reply(""), "")
        self.assertEqual(parse_judge_reply(None), "")
        self.assertEqual(parse_judge_reply("banana"), "")


class TestKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        self.assertEqual(cohen_kappa(["R", "C", "R"], ["R", "C", "R"]), 1.0)

    def test_known_value(self):
        # 50% agreement that is pure chance: kappa = 0
        a = ["R", "R", "C", "C"]
        b = ["R", "C", "R", "C"]
        self.assertAlmostEqual(cohen_kappa(a, b), 0.0)
        # po=0.75, pe=0.5 -> kappa = 0.5
        self.assertAlmostEqual(
            cohen_kappa(["R", "R", "R", "C"], ["R", "R", "C", "C"]), 0.5)

    def test_degenerate_all_same(self):
        self.assertNotEqual(cohen_kappa(["R"] * 4, ["R"] * 4),
                            cohen_kappa(["R"] * 4, ["R"] * 4))  # nan != nan

    def test_length_mismatch(self):
        self.assertNotEqual(cohen_kappa(["R"], ["R", "C"]), 0.0)  # nan


class TestBinarize(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(binarize("REFUSAL", "REFUSAL"), "R")
        self.assertEqual(binarize("COMPLY-UNREVIEWED", "REFUSAL"), "C")
        self.assertIsNone(binarize("EMPTY", "REFUSAL"))
        self.assertIsNone(binarize("JUDGE-UNPARSEABLE", "REFUSAL"))


class TestJudgeRowInfraPassthrough(unittest.TestCase):
    """Infra rows must never hit the network."""

    def test_infra_short_circuit(self):
        for lab in ("REQUEST-ERROR", "EMPTY", "DRY-RUN"):
            self.assertEqual(
                judge_row({"label": lab, "response": ""}, "m",
                          "http://127.0.0.1:1/v1"),
                lab)
        self.assertEqual(
            judge_row({"label": "COMPLY-UNREVIEWED",
                       "response": "REQUEST-ERROR: boom"}, "m",
                      "http://127.0.0.1:1/v1"),
            "REQUEST-ERROR")


class TestAdjudicateFlags(unittest.TestCase):
    def test_late_refusal_detected(self):
        body = ("Sure, here is the information you asked for. " * 40 +
                "However, as an AI I must add a caveat.")
        late, engage = head_and_tail_flags(body)
        self.assertEqual(late, 1)
        self.assertEqual(engage, 1)

    def test_early_refusal_not_late(self):
        late, engage = head_and_tail_flags(
            "I'm sorry, but I cannot help with that.")
        self.assertEqual(late, 0)
        self.assertEqual(engage, 0)

    def test_error_rows_neutral(self):
        self.assertEqual(head_and_tail_flags("REQUEST-ERROR: x"), (0, 0))
        self.assertEqual(head_and_tail_flags(""), (0, 0))

    def test_enrich_priority_orders_by_overlap(self):
        hi, hi_p = enrich({"idx": "1", "label": "COMPLY-UNREVIEWED",
                           "decode_overlap": "0.9", "response": "DECODE: x"})
        lo, lo_p = enrich({"idx": "2", "label": "COMPLY-UNREVIEWED",
                           "decode_overlap": "0.0", "response": "hello"})
        self.assertGreater(hi_p, lo_p)
        self.assertEqual(hi["dec_line"], 1)
        self.assertEqual(lo["dec_line"], 0)

    def test_build_only_comply_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.csv")
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "idx", "prompt", "decode_overlap", "response", "label"])
                w.writeheader()
                w.writerow({"idx": 0, "prompt": "a", "decode_overlap": 0.0,
                            "response": "x", "label": "COMPLY-UNREVIEWED"})
                w.writerow({"idx": 1, "prompt": "b", "decode_overlap": 0.8,
                            "response": "DECODE: b", "label":
                            "COMPLY-UNREVIEWED"})
                w.writerow({"idx": 2, "prompt": "c", "decode_overlap": "",
                            "response": "no", "label": "REFUSAL"})
            rows = build(p)
            self.assertEqual([r["idx"] for r in rows], ["1", "0"])
            self.assertTrue(all("human_verdict" in r for r in rows))


class TestJudgePromptSafety(unittest.TestCase):
    def test_system_prompt_is_classification_only(self):
        low = JUDGE_SYSTEM.lower()
        for banned in ("decode", "translate", "rot13", "base64", "spun"):
            self.assertNotIn(banned, low)
        self.assertIn("one word", low)


if __name__ == "__main__":
    unittest.main(verbosity=2)
