import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from router.verify_billed_tokens import verify


class _FakeResponse:
    def __init__(self, prompt_tokens, strategy=None):
        self.prompt_tokens = prompt_tokens
        self.strategy = strategy

    def raise_for_status(self):
        return None

    def json(self):
        body = {"usage": {"prompt_tokens": self.prompt_tokens}}
        if self.strategy:
            body["token_tax"] = {"strategy_used": self.strategy}
        return body


class _FakeClient:
    requests = []

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        type(self).requests = []
        return self

    def __exit__(self, *args):
        return None

    def post(self, url, headers=None, json=None):
        type(self).requests.append({"url": url, "headers": headers, "json": json})
        if url.startswith("https://provider.test"):
            return _FakeResponse(11)
        return _FakeResponse(8, "romanized")


class BilledVerificationTests(unittest.TestCase):
    def test_selects_kannada_and_saves_provider_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples_path = root / "examples.csv"
            output_path = root / "usage.csv"
            with examples_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["id", "lang", "text"])
                writer.writeheader()
                writer.writerow({"id": "english", "lang": "en", "text": "skip this"})
                for index in range(50):
                    writer.writerow(
                        {"id": str(index), "lang": "kn", "text": f"prompt {index}"}
                    )

            with patch(
                "router.verify_billed_tokens.httpx.Client", _FakeClient
            ), redirect_stdout(io.StringIO()):
                verify(
                    examples_path,
                    output_path,
                    50,
                    "test-model",
                    "https://provider.test/v1",
                    "http://127.0.0.1:8000",
                    "test-key",
                    "proxy-key",
                    "kn",
                )

            self.assertEqual(len(_FakeClient.requests), 100)
            self.assertEqual(_FakeClient.requests[0]["json"]["messages"][0]["content"], "prompt 0")
            self.assertEqual(_FakeClient.requests[1]["json"]["language"], "kn")
            self.assertEqual(
                _FakeClient.requests[1]["headers"]["Authorization"],
                "Bearer proxy-key",
            )
            with output_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 50)
            self.assertTrue(all(row["lang"] == "kn" for row in rows))
            self.assertTrue(all(row["tokens_saved"] == "3" for row in rows))
            self.assertTrue(all(row["strategy_used"] == "romanized" for row in rows))


if __name__ == "__main__":
    unittest.main()
