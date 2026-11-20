import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from audit.app import (
    LANGUAGE_SAMPLES,
    convert_text,
    count_text,
    select_language_sample,
)
from transform.romanize import romanize_text
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "audit/app.py"


class ConversionTests(unittest.TestCase):
    def test_kannada_romanizes_using_member_b_implementation(self):
        source = "ನಮಸ್ಕಾರ"
        result = romanize_text(source, "kn", "ISO")

        self.assertTrue(result)
        self.assertNotEqual(result, source)
        self.assertNotIn("ನ", result)

    def test_live_token_count_table_uses_all_available_tokenizers(self):
        results = count_text("hello", {"small": lambda text: len(text), "big": lambda text: 10})

        self.assertEqual(results.to_dict("records"), [
            {"Tokenizer": "small", "Tokens": 5},
            {"Tokenizer": "big", "Tokens": 10},
        ])

    def test_native_strategy_returns_text_unchanged(self):
        source = "ನಮಸ್ಕಾರ"
        self.assertEqual(convert_text(source, "kn", "native"), source)

    def test_romanized_strategy_changes_script_not_meaning(self):
        source = "ನಮಸ್ಕಾರ"
        output = convert_text(source, "kn", "romanized")

        self.assertNotEqual(output, source)
        self.assertNotIn("ನ", output)

    def test_pivot_is_not_a_romanization_fallback(self):
        with self.assertRaisesRegex(ValueError, "translation worker"):
            convert_text("ನಮಸ್ಕಾರ", "kn", "pivot")

    def test_switching_language_replaces_only_the_untouched_sample(self):
        kannada = LANGUAGE_SAMPLES["kn"]
        self.assertEqual(
            select_language_sample("hi", kannada, kannada),
            LANGUAGE_SAMPLES["hi"],
        )
        self.assertEqual(
            select_language_sample("hi", "my own text", kannada),
            "my own text",
        )

    def test_ui_switching_to_hindi_updates_sample_and_mode_output(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=45)
        app.selectbox(key="converter_language").set_value("hi")
        app.selectbox(key="converter_strategy").set_value("romanized").run(timeout=45)

        source = app.text_area(key="converter_source").value
        self.assertIn("नमस्ते", source)
        self.assertNotIn("न", app.code[0].value)
        self.assertEqual(app.code[0].value, romanize_text(source, "hi", "ISO"))

        app.selectbox(key="converter_strategy").set_value("native").run(timeout=45)
        self.assertEqual(app.code[0].value, source)

    def test_pivot_mode_submits_translation_in_background(self):
        class FakeExecutor:
            def __init__(self):
                self.submitted = None

            def submit(self, function, text, language):
                self.submitted = (function, text, language)
                future = Future()
                future.set_result(("Hello, how are you?", "cpu"))
                return future

        executor = FakeExecutor()
        with patch("audit.app.get_translation_executor", return_value=executor):
            from audit.app import submit_translation

            future = submit_translation("kn", "ನಮಸ್ಕಾರ")

        self.assertEqual(future.result(), ("Hello, how are you?", "cpu"))
        self.assertEqual(executor.submitted[1:], ("ನಮಸ್ಕಾರ", "kn"))


if __name__ == "__main__":
    unittest.main()
