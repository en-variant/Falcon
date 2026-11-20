"""Compare direct and routed billed prompt tokens for a small prompt sample."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import httpx


def _prompt_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Each verification prompt must have non-empty text.")
    return value


def _usage(response: httpx.Response, label: str) -> tuple[int, dict[str, Any]]:
    response.raise_for_status()
    try:
        body = response.json()
    except ValueError as error:
        raise ValueError(f"{label} endpoint returned invalid JSON.") from error
    usage = body.get("usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("prompt_tokens"), int):
        raise ValueError(f"{label} response has no integer usage.prompt_tokens.")
    return usage["prompt_tokens"], body


def verify(
    examples_path: Path,
    output_path: Path,
    limit: int,
    model: str,
    direct_base_url: str,
    proxy_url: str,
    api_key: str,
    language: str,
) -> None:
    if not 50 <= limit <= 100:
        raise ValueError("--limit must be between 50 and 100.")
    if language not in {"en", "kn", "hi", "ta"}:
        raise ValueError("--lang must be one of en, kn, hi, or ta.")
    with examples_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"id", "lang", "text"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{examples_path} must contain columns {sorted(required)}.")
        prompts = []
        for row in reader:
            if row.get("lang") != language:
                continue
            prompts.append(row)
            if len(prompts) == limit:
                break
    if len(prompts) < limit:
        raise ValueError(f"{examples_path} has only {len(prompts)} prompts; need {limit}.")

    direct_url = f"{direct_base_url.rstrip('/')}/chat/completions"
    proxy_url = f"{proxy_url.rstrip('/')}/v1/chat/completions"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "lang",
        "direct_prompt_tokens",
        "proxy_prompt_tokens",
        "tokens_saved",
        "strategy_used",
    ]
    with (
        httpx.Client(timeout=120.0) as client,
        output_path.open("w", newline="", encoding="utf-8") as destination,
    ):
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        destination.flush()
        for row in prompts:
            text = _prompt_text(row.get("text"))
            language = row.get("lang", "")
            messages = [{"role": "user", "content": text}]
            body = {"model": model, "messages": messages, "temperature": 0}
            direct_response = client.post(
                direct_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
            direct_tokens, _ = _usage(direct_response, "Direct")
            proxy_response = client.post(
                proxy_url,
                json={**body, "language": language},
            )
            proxy_tokens, proxy_body = _usage(proxy_response, "Proxy")
            token_tax = proxy_body.get("token_tax")
            strategy = token_tax.get("strategy_used") if isinstance(token_tax, dict) else ""
            writer.writerow(
                {
                    "id": row["id"],
                    "lang": language,
                    "direct_prompt_tokens": direct_tokens,
                    "proxy_prompt_tokens": proxy_tokens,
                    "tokens_saved": direct_tokens - proxy_tokens,
                    "strategy_used": strategy,
                }
            )
            destination.flush()
            print(
                f"Completed {row['id']}: direct={direct_tokens}, "
                f"proxy={proxy_tokens}, strategy={strategy or 'not reported'}"
            )
    print(f"Saved billed usage rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=Path("data/eval/examples.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/router/billed_token_verification.csv"),
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--lang", choices=("en", "kn", "hi", "ta"), default="kn")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    parser.add_argument(
        "--direct-base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--proxy-url", default=os.getenv("TOKEN_TAX_PROXY_URL", "http://127.0.0.1:8000")
    )
    args = parser.parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        parser.error("Set OPENAI_API_KEY before running billed verification.")
    if not args.model:
        parser.error("Set OPENAI_MODEL or pass --model.")
    verify(
        args.examples,
        args.output,
        args.limit,
        args.model,
        args.direct_base_url,
        args.proxy_url,
        api_key,
        args.lang,
    )


if __name__ == "__main__":
    main()
