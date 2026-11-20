import csv
import json
import tempfile
import unittest
from pathlib import Path

from audit.count_tokens import (
    LANGUAGES,
    count_tokens,
    load_pricing,
    load_sentences,
    write_csv,
)


def create_flores_files(folder: Path, language_rows: dict[str, list[str]]) -> None:
    folder.mkdir(parents=True)
    for language, flores_code in LANGUAGES.items():
        (folder / f"{flores_code}.devtest").write_text(
            "\n".join(language_rows[language]) + "\n",
            encoding="utf-8",
        )



class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def make_rows(self):
        return {
            language: [f"{language} one", f"{language} two"]
            for language in LANGUAGES
        }

    def test_load_sentences_accepts_aligned_nonempty_files(self):
        rows = self.make_rows()
        folder = self.root / "devtest"
        create_flores_files(folder, rows)

        self.assertEqual(load_sentences(folder, expected_count=2), rows)

    def test_load_sentences_rejects_wrong_line_count(self):
        rows = {language: [f"{language} one"] for language in LANGUAGES}
        folder = self.root / "devtest"
        create_flores_files(folder, rows)

        with self.assertRaisesRegex(ValueError, "expected 2"):
            load_sentences(folder, expected_count=2)

    def test_load_sentences_rejects_blank_line(self):
        rows = self.make_rows()
        rows["kn"][1] = " "
        folder = self.root / "devtest"
        create_flores_files(folder, rows)

        with self.assertRaisesRegex(ValueError, "empty line"):
            load_sentences(folder, expected_count=2)

    def test_count_tokens_uses_configured_prices_only(self):
        rows = count_tokens(
            {"en": ["hello"], "kn": ["ನಮಸ್ಕಾರ"]},
            {"sample": len},
            usd_inr=80,
            prices={"sample": (2.5, "priced-model")},
        )

        self.assertEqual(
            [(row["id"], row["lang"], row["tokenizer"], row["tokens"]) for row in rows],
            [(0, "en", "sample", 5), (0, "kn", "sample", 7)],
        )
        self.assertAlmostEqual(rows[0]["cost_inr"], 0.001)
        self.assertAlmostEqual(rows[1]["cost_inr"], 0.0014)

    def test_unpriced_tokenizers_have_blank_cost_and_csv_contract(self):
        rows = count_tokens({"en": ["hello"]}, {"local": len}, 80, {})
        output = self.root / "counts.csv"
        write_csv(rows, output)

        with output.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            self.assertEqual(
                reader.fieldnames, ["id", "lang", "tokenizer", "tokens", "cost_inr"]
            )
            self.assertEqual(
                next(reader),
                {
                    "id": "0",
                    "lang": "en",
                    "tokenizer": "local",
                    "tokens": "5",
                    "cost_inr": "",
                },
            )

    def test_load_pricing_rejects_invalid_exchange_rate(self):
        pricing_file = self.root / "pricing.json"
        pricing_file.write_text(
            json.dumps(
                {
                    "usd_inr": {"rate": 0},
                    "benchmarks": {},
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Invalid USD-INR rate"):
            load_pricing(pricing_file)
