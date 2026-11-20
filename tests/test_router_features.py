import unittest

from router.features import extract_features, infer_language


class FeatureTests(unittest.TestCase):
    def test_empty_input_has_finite_zero_features(self):
        features = extract_features("")
        self.assertEqual(features["char_length"], 0)
        self.assertEqual(features["word_count"], 0)
        self.assertTrue(all(value == 0 for value in features.values()))

    def test_indic_and_latin_ratios_detect_code_mixing(self):
        features = extract_features("ನಮಸ್ಕಾರ hello 42!")
        self.assertGreater(features["indic_proportion"], 0)
        self.assertGreater(features["latin_proportion"], 0)
        self.assertGreater(features["digit_proportion"], 0)
        self.assertGreater(features["punctuation_proportion"], 0)
        self.assertEqual(infer_language("ನಮಸ್ಕಾರ"), "kn")

    def test_urls_are_counted(self):
        features = extract_features("Read https://example.com now")
        self.assertGreater(features["url_proportion"], 0)

    def test_non_indic_text_defaults_to_english(self):
        self.assertEqual(infer_language("Hello, world!"), "en")


if __name__ == "__main__":
    unittest.main()
