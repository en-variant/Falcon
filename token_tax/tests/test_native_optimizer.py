import importlib.util
import json
import unittest
from pathlib import Path

import tiktoken

import native_language_optimizer as optimizer


ROOT = Path(__file__).resolve().parents[1]


def load_dashboard():
    spec = importlib.util.spec_from_file_location(
        "native_optimizer_dashboard",
        ROOT / "4_dashboard.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load dashboard module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeUserOptimizerTests(unittest.TestCase):
    def test_public_optimizer_returns_requested_sentence_and_change_fields(self):
        source = "  \u0c95\u0ca8\u0ccd\u0ca8\u0ca1   \u0caa\u0cb0\u0cc0\u0c95\u0ccd\u0cb7\u0cc6  "
        result = optimizer.optimize_native_text(source, "kn", "cl100k_base")

        self.assertEqual(result["original_sentence"], source)
        self.assertEqual(
            result["optimized_sentence"],
            "\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \u0caa\u0cb0\u0cc0\u0c95\u0ccd\u0cb7\u0cc6",
        )
        self.assertEqual(result["optimization_status"], "optimized")
        self.assertEqual(result["repeated_words_detected"], [])
        self.assertEqual(result["repeated_words_removed"], [])
        self.assertTrue(result["whitespace_changes"])

    def test_normalizes_extra_whitespace_and_compacts_excess_blank_lines(self):
        source = "  ಕನ್ನಡ\t\tಪರೀಕ್ಷೆ  \n\n\n ಇನ್ನೊಂದು ಸಾಲು   "
        result = optimizer.optimize_user_text(source, "kn", "cl100k_base")

        self.assertEqual(result["optimized_text"], "ಕನ್ನಡ ಪರೀಕ್ಷೆ\n\nಇನ್ನೊಂದು ಸಾಲು")
        self.assertLessEqual(result["optimized_tokens"], result["original_tokens"])
        self.assertIn("formatting_whitespace", {change["type"] for change in result["changes"]})

    def test_applies_nfc_canonical_normalization(self):
        result = optimizer.optimize_user_text("  cafe\u0301  ", "en", "cl100k_base")

        self.assertEqual(result["optimized_text"], "café")
        self.assertIn("unicode_nfc", {change["type"] for change in result["changes"]})

    def test_duplicate_word_is_retained_until_user_confirms_it(self):
        source = "ನಮಸ್ಕಾರ ನಮಸ್ಕಾರ!"
        suggested = optimizer.optimize_user_text(source, "kn", "cl100k_base")

        self.assertEqual(suggested["optimized_text"], source)
        self.assertEqual(suggested["status"], "review_required")
        self.assertEqual(suggested["candidates"][0]["word"], "ನಮಸ್ಕಾರ")
        self.assertTrue(suggested["candidates"][0]["requires_confirmation"])

        confirmed = optimizer.optimize_user_text(
            source,
            "kn",
            "cl100k_base",
            [suggested["candidates"][0]["id"]],
        )
        self.assertEqual(confirmed["optimized_text"], "ನಮಸ್ಕಾರ!")
        self.assertEqual(confirmed["changes"][0]["removed_text"], "ನಮಸ್ಕಾರ")
        self.assertTrue(confirmed["optimized_text"].endswith("!"))
        self.assertLessEqual(confirmed["optimized_tokens"], confirmed["original_tokens"])

    def test_user_example_is_cleaned_after_confirming_both_repeated_words(self):
        source = "ನನಗೆ      ನನಗೆ      ಕುಡಿಯಲು      ನೀರು      ಬೇಕು ಬೇಕು."
        proposal = optimizer.optimize_native_text(source, "kn", "cl100k_base")

        self.assertEqual(proposal["optimized_text"], "ನನಗೆ ನನಗೆ ಕುಡಿಯಲು ನೀರು ಬೇಕು ಬೇಕು.")
        self.assertEqual(proposal["repeated_words_detected"], ["ನನಗೆ", "ಬೇಕು"])
        self.assertEqual(proposal["optimization_status"], "review_required")
        confirmed = optimizer.optimize_native_text(
            source,
            "kn",
            "cl100k_base",
            [candidate["id"] for candidate in proposal["candidates"]],
        )

        self.assertEqual(confirmed["optimized_text"], "ನನಗೆ ಕುಡಿಯಲು ನೀರು ಬೇಕು.")
        self.assertEqual(confirmed["repeated_words_removed"], [
            change for change in confirmed["changes"] if change["type"] == "duplicate_word"
        ])

    def test_intentional_repetition_is_not_removed_automatically(self):
        source = "ಮತ್ತೆ ಮತ್ತೆ ಹೇಳಿ."
        result = optimizer.optimize_user_text(source, "kn", "o200k_base")

        self.assertEqual(result["optimized_text"], source)
        self.assertEqual(result["tokens_saved"], 0)
        self.assertEqual(result["status"], "review_required")

    def test_kannada_unicode_punctuation_and_selected_tokenizer_counts(self):
        source = "  \u0cad\u0cbe\u0cb0\u0ca4\u0ca6 \u0cb0\u0cbe\u0c9c\u0ca7\u0cbe\u0ca8\u0cbf \u0caf\u0cbe\u0cb5\u0cc1\u0ca6\u0cc1?  "
        result = optimizer.optimize_user_text(source, "kn", "p50k_base")
        encoder = tiktoken.get_encoding("p50k_base")

        self.assertEqual(result["optimized_text"], "ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?")
        self.assertTrue(result["optimized_text"].endswith("?"))
        self.assertEqual(
            result["original_tokens"],
            len(encoder.encode(source, disallowed_special=())),
        )
        self.assertEqual(
            result["optimized_tokens"],
            len(encoder.encode(result["optimized_text"], disallowed_special=())),
        )

    def test_all_supported_language_codes_accept_native_text(self):
        examples = {
            "kn": "ಕನ್ನಡ ಪರೀಕ್ಷೆ.",
            "ta": "தமிழ் சோதனை.",
            "te": "తెలుగు పరీక్ష.",
            "ml": "മലയാളം പരിശോധന.",
            "hi": "हिंदी परीक्षण।",
            "mr": "मराठी चाचणी.",
            "bn": "বাংলা পরীক্ষা।",
            "gu": "ગુજરાતી પરીક્ષણ.",
            "pa": "ਪੰਜਾਬੀ ਜਾਂਚ।",
            "or": "ଓଡ଼ିଆ ପରୀକ୍ଷା।",
            "ur": "اردو آزمائش۔",
            "en": "English test.",
        }
        for language, text in examples.items():
            with self.subTest(language=language):
                result = optimizer.optimize_user_text(text, language, "cl100k_base")
                self.assertEqual(result["optimized_text"], text)
                self.assertEqual(result["tokens_saved"], 0)

    def test_clean_text_reports_zero_savings_and_no_safe_change(self):
        source = "ಕನ್ನಡ ವಾಕ್ಯ."
        result = optimizer.optimize_user_text(source, "kn", "o200k_base")

        self.assertEqual(result["original_text"], result["optimized_text"])
        self.assertEqual(result["tokens_saved"], 0)
        self.assertEqual(result["savings_percentage"], 0)
        self.assertEqual(result["status"], "no_safe_optimization")

    def test_rejects_a_transformation_that_increases_selected_token_count(self):
        questions = json.loads(
            (ROOT / "indian_language_questions.json").read_text(encoding="utf-8")
        )["languages"]
        source = " " + questions["Hindi"][8] + " "
        encoder = tiktoken.get_encoding("o200k_base")
        proposed, _ = optimizer._format_text(source)
        self.assertGreater(
            len(encoder.encode(proposed, disallowed_special=())),
            len(encoder.encode(source, disallowed_special=())),
        )
        result = optimizer.optimize_user_text(source, "hi", "o200k_base")

        self.assertEqual(result["status"], "rejected_token_increase")
        self.assertEqual(result["optimized_text"], source)
        self.assertEqual(result["optimized_tokens"], result["original_tokens"])
        self.assertEqual(result["tokens_saved"], 0)

    def test_detects_invisibles_but_preserves_orthographic_joiners(self):
        removable_edge_space = optimizer.optimize_user_text("A\u200b B", "en", "cl100k_base")
        urdu_text = "\u0645\u06cc\u200c\u06ba"
        orthographic_joiner = optimizer.optimize_user_text(urdu_text, "ur", "cl100k_base")

        self.assertEqual(removable_edge_space["optimized_text"], "A B")
        self.assertTrue(any(
            item["codepoint"] == "U+200B" and item["action"] == "removed"
            for item in removable_edge_space["detected_invisible_characters"]
        ))
        self.assertEqual(orthographic_joiner["optimized_text"], urdu_text)
        self.assertTrue(any(
            item["codepoint"] == "U+200C" and item["action"] == "preserved"
            for item in orthographic_joiner["detected_invisible_characters"]
        ))

    def test_api_validates_required_fields_text_language_and_encoding(self):
        client = load_dashboard().app.test_client()
        requests = [
            ({}, 400),
            ({"text": "x", "language": "kn"}, 400),
            ({"text": "   ", "language": "kn", "encoding": "cl100k_base"}, 400),
            ({"text": "x", "language": "xx", "encoding": "cl100k_base"}, 400),
            ({"text": "x", "language": "kn", "encoding": "invalid"}, 400),
            ({"text": 4, "language": "kn", "encoding": "cl100k_base"}, 400),
            ({"text": "x", "language": "kn", "encoding": "cl100k_base",
              "confirmed_duplicate_ids": ["unknown"]}, 400),
        ]
        for payload, expected_status in requests:
            with self.subTest(payload=payload):
                response = client.post("/api/native-optimize", json=payload)
                self.assertEqual(response.status_code, expected_status)
                self.assertIn("error", response.get_json())

    def test_api_returns_required_metrics_for_real_kannada_input(self):
        client = load_dashboard().app.test_client()
        source = "  \u0cad\u0cbe\u0cb0\u0ca4\u0ca6 \u0cb0\u0cbe\u0c9c\u0ca7\u0cbe\u0ca8\u0cbf \u0caf\u0cbe\u0cb5\u0cc1\u0ca6\u0cc1?  "
        response = client.post("/api/native-optimize", json={
            "text": source,
            "language": "kn",
            "encoding": "o200k_base",
        })

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["original_text"], source)
        self.assertEqual(result["optimized_text"], "ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?")
        for field in (
            "original_tokens",
            "optimized_tokens",
            "tokens_saved",
            "savings_percentage",
            "changes",
            "status",
        ):
            self.assertIn(field, result)
        self.assertEqual(result["tokens_saved"], result["original_tokens"] - result["optimized_tokens"])
        self.assertFalse(result["semantic_equivalence_proven"])

    def test_api_removes_duplicate_only_after_candidate_confirmation(self):
        client = load_dashboard().app.test_client()
        source = "ನಮಸ್ಕಾರ ನಮಸ್ಕಾರ!"
        payload = {"text": source, "language": "kn", "encoding": "cl100k_base"}
        suggestion = client.post("/api/native-optimize", json=payload).get_json()

        self.assertEqual(suggestion["optimized_text"], source)
        self.assertEqual(suggestion["status"], "review_required")
        payload["confirmed_duplicate_ids"] = [suggestion["candidates"][0]["id"]]
        confirmed = client.post("/api/native-optimize", json=payload).get_json()

        self.assertEqual(confirmed["optimized_text"], "ನಮಸ್ಕಾರ!")
        self.assertEqual(confirmed["changes"][0]["type"], "duplicate_word")
        self.assertEqual(confirmed["changes"][0]["removed_text"], "ನಮಸ್ಕಾರ")

    def test_dashboard_contains_optimizer_controls(self):
        dashboard = load_dashboard()
        client = dashboard.app.test_client()
        page = client.get("/")

        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Native-language sentence optimizer", page.data)
        self.assertIn(b"native-optimizer-language", page.data)
        self.assertIn(b"native-optimizer-encoding", page.data)


if __name__ == "__main__":
    unittest.main()
