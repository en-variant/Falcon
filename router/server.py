"""OpenAI-compatible proxy endpoint with local strategy selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import joblib
import pandas as pd
import tiktoken
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from router.features import extract_features, infer_language
from router.transforms import apply_strategy


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    language: str | None = None


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
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


def _load_router(path: Path) -> tuple[Any, list[str]]:
    if not path.is_file():
        raise RuntimeError(
            f"Router model not found at {path}; run python -m router.train_eval first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
        raise RuntimeError(f"Router model bundle at {path} has an invalid format.")
    return bundle["model"], list(bundle["feature_columns"])


def create_app(model_path: Path | None = None) -> FastAPI:
    model_path = model_path or Path(
        os.getenv("TOKEN_TAX_ROUTER_MODEL", "router/model.joblib")
    )
    model, feature_columns = _load_router(model_path)
    app = FastAPI(title="Token Tax Optimizer", version="1.0.0")
    app.state.router_model = model
    app.state.feature_columns = feature_columns

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not supported by this proxy yet; set stream=false.",
            )
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty.")
        if request.language is not None and request.language not in {"en", "kn", "hi", "ta"}:
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
        features = extract_features(user_text)
        strategy = str(
            app.state.router_model.predict(
                pd.DataFrame(
                    [{key: features[key] for key in app.state.feature_columns}],
                    columns=app.state.feature_columns,
                )
            )[0]
        )
        language = request.language or infer_language(user_text)
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
            "token_count_basis": "cl100k_base text-only estimate",
            "billed_prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        }
        return result

    return app


app = create_app()
