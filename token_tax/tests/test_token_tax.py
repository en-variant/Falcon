import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TokenTaxTests(unittest.TestCase):
    def test_benchmark_measures_ten_samples_for_all_twelve_languages(self):
        benchmark = load_script("1_fertility_benchmarker.py", "test_benchmark")
        results = benchmark.run_benchmark()
        self.assertEqual(len(results), 12)
        self.assertEqual(results["Hindi"]["sample_count"], 10)
        self.assertEqual(
            set(results["Hindi"]["tokenizers"]),
            {"cl100k (GPT-4)", "o200k (GPT-4o)", "p50k (GPT-3)"},
        )
        self.assertEqual(len(results["Hindi"]["tokenizers"]["cl100k (GPT-4)"]["samples"]), 10)
        self.assertEqual(results["English"]["tokenizers"]["cl100k (GPT-4)"]["multiplier_vs_english"], 1)

    def test_word_count_keeps_combining_marks_with_word(self):
        benchmark = load_script("1_fertility_benchmarker.py", "test_word_count")
        self.assertEqual(benchmark.count_words("कि भारत, 2025!"), 3)

    def test_transliteration_reports_urdu_as_unsupported_and_measures_hindi(self):
        pipeline = load_script("2_transliteration_pipeline.py", "test_pipeline")
        results = pipeline.run_pipeline()
        self.assertEqual(results["Urdu"]["status"], "unsupported")
        self.assertIsNone(results["Urdu"]["savings_percent"])
        self.assertEqual(results["Hindi"]["status"], "measured")
        self.assertEqual(results["Hindi"]["sample_count"], 10)
        self.assertIsInstance(results["Hindi"]["samples"][0]["round_trip_exact"], bool)
        self.assertIsInstance(
            results["Hindi"]["samples"][0]["round_trip_character_similarity_percent"],
            float,
        )
        self.assertEqual(
            results["Hindi"]["samples"][0]["round_trip_character_similarity_percent"],
            100.0 if results["Hindi"]["samples"][0]["round_trip_exact"] else
            pipeline.character_similarity(
                results["Hindi"]["samples"][0]["original"],
                results["Hindi"]["samples"][0]["round_trip"],
            ),
        )
        with self.assertRaises(ValueError):
            pipeline.to_latin("اردو", "Urdu")

    def test_sentencepiece_uses_held_out_questions(self):
        vocabulary = load_script("3_vocab_expansion_demo.py", "test_vocab")
        report = vocabulary.run_vocab_expansion()
        self.assertEqual(report["experiment"]["training_sample_count"], 88)
        self.assertEqual(report["experiment"]["heldout_sample_count"], 22)
        self.assertEqual(report["experiment"]["exact_round_trips"], 22)
        self.assertIn("not a vocabulary merge", report["experiment"]["production_claim"])
        self.assertEqual(
            report["languages"]["Hindi"]["samples"][0]["id"],
            "q09",
        )
        custom = vocabulary.evaluate_custom_text("भारत में जल संरक्षण कैसे करें?")
        self.assertTrue(custom["round_trip_exact"])
        self.assertEqual(
            custom["sentencepiece_piece_count"],
            len(custom["sentencepiece_pieces"]),
        )
        self.assertGreater(custom["cl100k_token_count"], 0)
        self.assertFalse(custom["semantic_or_answer_quality_evaluated"])
        self.assertIn("not a GPT or Gemini vocabulary", custom["tokenizer"])
        with self.assertRaises(ValueError):
            vocabulary.evaluate_custom_text("x" * 100_001)

    def test_dashboard_returns_explicit_transliteration_availability(self):
        dashboard = load_script("4_dashboard.py", "test_dashboard")
        client = dashboard.app.test_client()
        hindi = client.post("/api/inspect", json={"text": "भारत क्या है?"})
        urdu = client.post("/api/inspect", json={"text": "بھارت کیا ہے؟"})
        invalid = client.post("/api/inspect", json={"text": 12})
        self.assertEqual(hindi.status_code, 200)
        self.assertTrue(hindi.json["latin"]["available"])
        self.assertEqual(urdu.status_code, 200)
        self.assertFalse(urdu.json["latin"]["available"])
        self.assertIn("language not inferred", hindi.json["detected_script"])
        self.assertEqual(invalid.status_code, 400)
        custom = client.post("/api/vocab/inspect", json={"text": "தமிழ் சோதனை"})
        self.assertEqual(custom.status_code, 200)
        self.assertTrue(custom.json["round_trip_exact"])
        self.assertEqual(
            custom.json["sentencepiece_piece_count"],
            len(custom.json["sentencepiece_pieces"]),
        )
        self.assertEqual(
            client.post("/api/vocab/inspect", json={"text": 4}).status_code,
            400,
        )
        self.assertEqual(
            client.post("/api/vocab/inspect", json={"text": "x" * 100_001}).status_code,
            413,
        )
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Sentence optimizer", page.data)
        self.assertIn(b"Evaluate custom text with the trained SentencePiece tokenizer", page.data)


if __name__ == "__main__":
    unittest.main()
