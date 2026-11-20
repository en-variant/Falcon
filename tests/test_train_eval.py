import tempfile
import unittest
from pathlib import Path

import pandas as pd

from router.train_eval import choose_target, evaluate, generate_synthetic


class RouterTrainingTests(unittest.TestCase):
    def test_target_is_cheapest_correct_or_native_if_none_correct(self):
        results = pd.DataFrame(
            [
                {"strategy": "native", "correct": 1, "tokens": 12},
                {"strategy": "romanized", "correct": 1, "tokens": 8},
                {"strategy": "pivot", "correct": 0, "tokens": 4},
            ]
        )
        self.assertEqual(choose_target(results), "romanized")
        results["correct"] = 0
        self.assertEqual(choose_target(results), "native")

    def test_synthetic_pipeline_respects_group_split_and_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "results.csv"
            examples_path = root / "examples.csv"
            model_path = root / "router.joblib"
            predictions_path = root / "predictions.csv"

            generate_synthetic(results_path, examples_path)
            results = pd.read_csv(results_path)
            examples = pd.read_csv(examples_path)
            self.assertEqual(len(results), 300)
            metrics = {
                row["strategy"]: row
                for row in evaluate(
                    results, examples, model_path, predictions_path
                )
            }

            self.assertEqual(metrics["router"]["accuracy"], 1.0)
            self.assertEqual(
                metrics["router"]["mean_tokens"],
                metrics["oracle"]["mean_tokens"],
            )
            predictions = pd.read_csv(predictions_path)
            self.assertEqual(list(predictions.columns), ["id", "chosen_strategy"])
            self.assertTrue(model_path.is_file())


if __name__ == "__main__":
    unittest.main()
