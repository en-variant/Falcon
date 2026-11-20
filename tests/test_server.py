import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import pandas as pd

from router.train_eval import evaluate, generate_synthetic


class _FakeResponse:
    is_error = False

    def json(self):
        return {
            "id": "completion-test",
            "choices": [],
            "usage": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7},
        }


class _FakeAsyncClient:
    captured_request = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, headers, json):
        type(self).captured_request = {"url": url, "headers": headers, "json": json}
        return _FakeResponse()


class ServerTests(unittest.TestCase):
    def test_native_request_reaches_upstream_unchanged_and_reports_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "results.csv"
            examples_path = root / "examples.csv"
            model_path = root / "model.joblib"
            generate_synthetic(results_path, examples_path)
            evaluate(
                pd.read_csv(results_path),
                pd.read_csv(examples_path),
                model_path,
                root / "predictions.csv",
                bootstrap_samples=100,
                training_data_kind="synthetic",
            )
            with patch.dict(
                os.environ,
                {
                    "TOKEN_TAX_ROUTER_MODEL": str(model_path),
                    "TOKEN_TAX_USAGE_DB": str(root / "usage.sqlite3"),
                    "TOKEN_TAX_DAILY_REQUEST_LIMIT": "1",
                },
            ):
                from router.server import create_app
                app = create_app(model_path, allow_non_real_model=True)

            original_messages = [
                {"role": "system", "content": "Answer with one word."},
                {"role": "user", "content": "hello short 0"},
            ]
            with (
                patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "test-key",
                        "TOKEN_TAX_PROXY_API_KEY": "proxy-key",
                    },
                ),
                patch("router.server.httpx.AsyncClient", _FakeAsyncClient),
                TestClient(app) as client,
            ):
                denied = client.post(
                    "/v1/chat/completions",
                    json={"model": "test-model", "messages": original_messages},
                )
                response = client.post(
                    "/v1/chat/completions",
                    json={"model": "test-model", "messages": original_messages},
                    headers={"Authorization": "Bearer proxy-key"},
                )
                over_limit = client.post(
                    "/v1/chat/completions",
                    json={"model": "test-model", "messages": original_messages},
                    headers={"Authorization": "Bearer proxy-key"},
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(over_limit.status_code, 429)
        self.assertEqual(
            _FakeAsyncClient.captured_request["json"]["messages"], original_messages
        )
        self.assertEqual(response.json()["token_tax"]["strategy_used"], "native")
        self.assertEqual(response.json()["token_tax"]["billed_prompt_tokens"], 6)

    def test_synthetic_model_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path = root / "results.csv"
            examples_path = root / "examples.csv"
            model_path = root / "model.joblib"
            generate_synthetic(results_path, examples_path)
            evaluate(
                pd.read_csv(results_path),
                pd.read_csv(examples_path),
                model_path,
                root / "predictions.csv",
                bootstrap_samples=100,
                training_data_kind="synthetic",
            )
            from router.server import create_app

            app = create_app(model_path)
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").json()["status"], "not_ready")


if __name__ == "__main__":
    unittest.main()
