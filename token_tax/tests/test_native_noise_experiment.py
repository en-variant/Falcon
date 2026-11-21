import unittest

import native_noise_experiment as experiment


class NativeNoiseExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = experiment.build_dataset()
        cls.report = experiment.run_experiment()

    def test_dataset_has_controls_and_four_variants_per_source_prompt(self):
        records = self.dataset["records"]
        self.assertEqual(len(records), 550)
        self.assertEqual(sum(row["condition"] == "clean_control" for row in records), 110)
        self.assertEqual(sum(row["synthetic_augmentation"] for row in records), 440)
        for language in experiment.CONTEXT_PREFIXES:
            language_rows = [row for row in records if row["language"] == language]
            self.assertEqual(len(language_rows), 50)
            self.assertEqual(
                sum(row["condition"] == "clean_control" for row in language_rows),
                10,
            )

    def test_originals_are_retained_and_repeat_removal_is_not_applied(self):
        for row in self.report["results"]:
            self.assertTrue(row["clean_control_query"])
            self.assertEqual(
                row["techniques"]["repetition_detection_only"]["optimized_query"],
                row["original_query"],
            )
            self.assertEqual(
                row["techniques"]["repetition_detection_only"]["semantic_evaluation"],
                "not_run",
            )
        self.assertEqual(
            self.report["experiment"]["techniques"]["repetition_detection_only"]
            ["overall"]["repetition_candidates_detected"],
            220,
        )

    def test_safe_normalizers_reduce_injected_whitespace_and_leave_clean_control(self):
        rows = self.report["results"]
        noisy = next(row for row in rows if row["condition"] == "whitespace_noise")
        formatted = next(row for row in rows if row["condition"] == "formatting_noise")
        control = next(row for row in rows if row["condition"] == "clean_control")
        normalized = noisy["techniques"]["horizontal_whitespace"]
        self.assertTrue(normalized["changed"])
        self.assertGreater(
            normalized["token_counts"]["cl100k_base"]["saved_tokens"],
            0,
        )
        self.assertGreater(
            formatted["techniques"]["blank_line_compaction"]["token_counts"]
            ["cl100k_base"]["saved_tokens"],
            formatted["techniques"]["outer_whitespace_trim"]["token_counts"]
            ["cl100k_base"]["saved_tokens"],
        )
        self.assertFalse(control["techniques"]["combined_safe_normalization"]["changed"])
        self.assertNotIn(
            "\r",
            formatted["techniques"]["blank_line_compaction"]["optimized_query"],
        )
        self.assertTrue(
            all(
                result["textual_invariant_passed"]
                for row in rows
                for result in row["techniques"].values()
            )
        )

    def test_results_include_per_language_and_per_condition_metrics(self):
        technique = self.report["experiment"]["techniques"]["combined_safe_normalization"]
        self.assertEqual(set(technique["by_language"]), set(experiment.CONTEXT_PREFIXES))
        self.assertEqual(
            set(technique["by_condition"]),
            {
                "clean_control",
                "whitespace_noise",
                "formatting_noise",
                "repeated_query",
                "redundant_context",
            },
        )
        self.assertEqual(
            set(technique["overall"]["by_tokenizer"]),
            set(experiment.TOKENIZERS),
        )

    def test_external_model_protocol_is_prepared_but_not_run(self):
        protocol = self.report["experiment"]["external_evaluation"]
        self.assertEqual(protocol["status"], "prepared_not_run")
        self.assertTrue(protocol["same_model_required"])
        self.assertEqual(protocol["temperature"], 0)
        self.assertIsNone(protocol["model"])


if __name__ == "__main__":
    unittest.main()
