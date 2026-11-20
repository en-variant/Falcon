"""OpenAI-compatible proxy endpoint with local strategy selection."""

from __future__ import annotations

import os
import logging
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import joblib
import pandas as pd
import tiktoken
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from router.features import (
    SUPPORTED_LANGUAGES,
    extract_routing_features,
    infer_language,
)
from router.transforms import apply_strategy

logger = logging.getLogger(__name__)
SUPPORTED_STRATEGIES = {"native", "romanized", "pivot"}


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    language: str | None = None


class DailyRequestLimitReached(Exception):
    """Raised when the process-wide daily upstream request budget is exhausted."""


class DailyRequestBudget:
    def __init__(self, path: Path, limit: int):
        if limit < 1:
            raise ValueError("TOKEN_TAX_DAILY_REQUEST_LIMIT must be positive.")
        self.path = path
        self.limit = limit
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def consume(self) -> None:
        day = datetime.now(timezone.utc).date().isoformat()
        with sqlite3.connect(self.path, timeout=30) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS daily_requests "
                "(request_date TEXT PRIMARY KEY, request_count INTEGER NOT NULL)"
            )
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT request_count FROM daily_requests WHERE request_date = ?",
                (day,),
            ).fetchone()
            used = int(row[0]) if row else 0
            if used >= self.limit:
                connection.rollback()
                raise DailyRequestLimitReached
            connection.execute(
                "INSERT INTO daily_requests(request_date, request_count) VALUES (?, 1) "
                "ON CONFLICT(request_date) DO UPDATE SET "
                "request_count = request_count + 1",
                (day,),
            )
            connection.commit()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _transform_message(
    message: dict[str, Any], strategy: str, language: str
) -> dict[str, Any]:
    if strategy == "native":
        return message
    updated = dict(message)
    content = message.get("content")
    if isinstance(content, str):
        updated["content"] = apply_strategy(content, strategy, language)
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                updated_part = dict(part)
                updated_part["text"] = apply_strategy(
                    str(part.get("text", "")), strategy, language
                )
                parts.append(updated_part)
            else:
                parts.append(part)
        updated["content"] = parts
    return updated


def _load_router(
    path: Path, allow_non_real_model: bool = False
) -> tuple[Any, list[str]]:
    if not path.is_file():
        raise RuntimeError(
            f"Router model not found at {path}; run python -m router.train_eval first."
        )
    try:
        bundle = joblib.load(path)
    except Exception as error:
        logger.exception("Could not load router model from %s", path)
        raise RuntimeError(
            f"Could not load router model at {path}: {type(error).__name__}."
        ) from error
    expected_features = list(extract_routing_features("", "en"))
    if (
        not isinstance(bundle, dict)
        or bundle.get("model_version") != 1
        or bundle.get("training_data_kind") not in {"real", "synthetic", "unverified"}
        or "model" not in bundle
        or bundle.get("feature_columns") != expected_features
    ):
        raise RuntimeError(f"Router model bundle at {path} has an invalid format.")
    if bundle["training_data_kind"] != "real" and not allow_non_real_model:
        raise RuntimeError(
            f"Router model at {path} is marked {bundle['training_data_kind']!r}; "
            "retrain it with real experiment results before serving requests."
        )
    model = bundle["model"]
    model_depth = getattr(model, "max_depth", None)
    model_classes = getattr(model, "classes_", ())
    if (
        not callable(getattr(model, "predict", None))
        or not isinstance(model_depth, int)
        or model_depth > 4
        or len(model_classes) == 0
        or not set(model_classes).issubset(SUPPORTED_STRATEGIES)
    ):
        raise RuntimeError(f"Router model at {path} violates the model constraints.")
    return model, expected_features


def create_app(
    model_path: Path | None = None, allow_non_real_model: bool = False
) -> FastAPI:
    model_path = model_path or Path(
        os.getenv("TOKEN_TAX_ROUTER_MODEL", "router/model.joblib")
    )
    app = FastAPI(title="Token Tax Optimizer", version="1.0.0")
    try:
        model, feature_columns = _load_router(model_path, allow_non_real_model)
        app.state.router_model = model
        app.state.feature_columns = feature_columns
        app.state.router_error = None
    except RuntimeError as error:
        app.state.router_model = None
        app.state.feature_columns = []
        app.state.router_error = str(error)
    daily_limit = int(os.getenv("TOKEN_TAX_DAILY_REQUEST_LIMIT", "100"))
    usage_path = Path(
        os.getenv("TOKEN_TAX_USAGE_DB", "data/router/request_usage.sqlite3")
    )
    app.state.request_budget = DailyRequestBudget(usage_path, daily_limit)

    @app.get("/health")
    async def health() -> dict[str, str]:
        if app.state.router_model is None:
            return {"status": "not_ready", "detail": app.state.router_error}
        missing_configuration = [
            name
            for name in (
                "OPENAI_API_KEY",
                "TOKEN_TAX_PROXY_API_KEY",
                "TOKEN_TAX_TRANSFORM_MODULE",
            )
            if not os.getenv(name)
        ]
        if missing_configuration:
            return {
                "status": "not_ready",
                "detail": f"Missing configuration: {', '.join(missing_configuration)}",
            }
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest, http_request: Request
    ) -> dict[str, Any]:
        proxy_api_key = os.getenv("TOKEN_TAX_PROXY_API_KEY")
        if not proxy_api_key:
            raise HTTPException(
                status_code=503, detail="TOKEN_TAX_PROXY_API_KEY is not configured."
            )
        expected_authorization = f"Bearer {proxy_api_key}"
        provided_authorization = http_request.headers.get("authorization", "")
        if not secrets.compare_digest(provided_authorization, expected_authorization):
            raise HTTPException(status_code=401, detail="Invalid proxy authorization.")
        if app.state.router_model is None:
            raise HTTPException(status_code=503, detail=app.state.router_error)
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not supported by this proxy yet; set stream=false.",
            )
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty.")
        if request.language is not None and request.language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400, detail="language must be one of en, kn, hi, or ta."
            )

        user_text = "\n".join(
            _message_text(message.get("content"))
            for message in request.messages
            if message.get("role") == "user"
        )
        if not user_text:
            raise HTTPException(
                status_code=400, detail="At least one user message with text is required."
            )
        language = request.language or infer_language(user_text)
        features = extract_routing_features(user_text, language)
        strategy = str(
            app.state.router_model.predict(
                pd.DataFrame(
                    [{key: features[key] for key in app.state.feature_columns}],
                    columns=app.state.feature_columns,
                )
            )[0]
        )
        if strategy not in SUPPORTED_STRATEGIES:
            raise HTTPException(status_code=503, detail="Router predicted an unsupported strategy.")
        transformed_messages = []
        for message in request.messages:
            if message.get("role") == "user":
                try:
                    transformed_messages.append(
                        _transform_message(message, strategy, language)
                    )
                except (RuntimeError, ValueError, TypeError) as error:
                    raise HTTPException(status_code=503, detail=str(error)) from error
            else:
                transformed_messages.append(message)

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")

        try:
            app.state.request_budget.consume()
        except DailyRequestLimitReached as error:
            raise HTTPException(
                status_code=429,
                detail="Daily request budget reached; try again after the UTC day resets.",
            ) from error
        except sqlite3.Error as error:
            logger.exception("Could not update daily API request budget")
            raise HTTPException(
                status_code=503, detail="Could not verify daily request budget."
            ) from error

        upstream_body = request.model_dump(exclude={"language"})
        upstream_body["messages"] = transformed_messages
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=upstream_body,
                )
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=502, detail=f"Upstream request failed: {error}"
            ) from error
        if response.is_error:
            raise HTTPException(
                status_code=response.status_code,
                detail={"upstream_error": response.text},
            )
        try:
            result = response.json()
        except ValueError as error:
            raise HTTPException(
                status_code=502, detail="Upstream returned invalid JSON."
            ) from error

        encoding = tiktoken.get_encoding("cl100k_base")
        original_tokens = len(encoding.encode(user_text))
        optimized_text = "\n".join(
            _message_text(message.get("content"))
            for message in transformed_messages
            if message.get("role") == "user"
        )
        optimized_tokens = len(encoding.encode(optimized_text))
        usage = result.get("usage")
        result["token_tax"] = {
            "strategy_used": strategy,
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "estimated_user_tokens_saved": original_tokens - optimized_tokens,
            "token_count_basis": "cl100k_base text-only estimate",
            "billed_prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        }
        return result

    return app


app = create_app()
