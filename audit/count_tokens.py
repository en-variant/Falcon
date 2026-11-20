"""Count tokenizer pieces in the aligned FLORES-200 devtest sentences."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import tiktoken


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {
    "en": "eng_Latn",
    "kn": "kan_Knda",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
}
EXPECTED_SENTENCES = 1012
HF_MODELS = {
    "qwen": (
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
    ),
    "sarvam": (
        "sarvamai/sarvam-1",
        "e9607337286ddf496d4a2562b194e489dcf3feea",
    ),
}

TokenCounter = Callable[[str], int]


def load_sentences(
    folder: Path, expected_count: int = EXPECTED_SENTENCES
) -> dict[str, list[str]]:
    sentences: dict[str, list[str]] = {}
    for language, flores_code in LANGUAGES.items():
        path = folder / f"{flores_code}.devtest"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Download the official FLORES-200 archive; "
                "see audit/README.md."
            )
        with path.open(encoding="utf-8") as source:
            lines = [line.rstrip("\r\n") for line in source]
        if len(lines) != expected_count:
            raise ValueError(
                f"{path} has {len(lines)} lines; expected {expected_count}."
            )
        if any(not line.strip() for line in lines):
            raise ValueError(f"{path} contains an empty line.")
        sentences[language] = lines

    counts = {len(rows) for rows in sentences.values()}
    if len(counts) != 1:
        raise ValueError("FLORES sentence files are not line-aligned.")
    return sentences


def build_tokenizers(offline: bool = False) -> dict[str, TokenCounter]:
    tokenizers: dict[str, TokenCounter] = {}
    for name in ("cl100k", "o200k"):
        encoding = tiktoken.get_encoding(f"{name}_base")
        tokenizers[name] = lambda text, enc=encoding: len(enc.encode(text))

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install the audit dependencies with `python -m pip install -r requirements.txt`."
        ) from error

    for label, (model_id, revision) in HF_MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            local_files_only=offline,
        )
        tokenizers[label] = lambda text, tok=tokenizer: len(
            tok.encode(text, add_special_tokens=False)
        )
    return tokenizers


def load_pricing(path: Path) -> tuple[float, dict[str, tuple[float, str]]]:
    with path.open(encoding="utf-8") as source:
        pricing = json.load(source)
    rate = float(pricing["usd_inr"]["rate"])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"Invalid USD-INR rate in {path}.")
    benchmarks: dict[str, tuple[float, str]] = {}
    for tokenizer_name, entry in pricing["benchmarks"].items():
        price = float(entry["usd_per_million_input_tokens"])
        if not math.isfinite(price) or price < 0:
            raise ValueError(f"Invalid input price for {tokenizer_name} in {path}.")
        benchmarks[tokenizer_name] = (price, entry["api_model"])
    return rate, benchmarks


def count_tokens(
    sentences: dict[str, list[str]],
    tokenizers: dict[str, TokenCounter],
    usd_inr: float,
    prices: dict[str, tuple[float, str]],
) -> list[dict[str, object]]:
    rows = []
    for language, texts in sentences.items():
        for sentence_id, text in enumerate(texts):
            for tokenizer_name, count in tokenizers.items():
                tokens = count(text)
                price = prices.get(tokenizer_name)
                cost_inr = (
                    (tokens / 1_000_000) * price[0] * usd_inr
                    if price is not None
                    else ""
                )
                rows.append(
                    {
                        "id": sentence_id,
                        "lang": language,
                        "tokenizer": tokenizer_name,
                        "tokens": tokens,
                        "cost_inr": cost_inr,
                    }
                )
    return rows


def write_csv(rows: Iterable[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["id", "lang", "tokenizer", "tokens", "cost_inr"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summaries(rows: list[dict[str, object]]) -> None:
    averages: dict[tuple[str, str], float] = {}
    for tokenizer_name in dict.fromkeys(str(row["tokenizer"]) for row in rows):
        for language in LANGUAGES:
            values = [
                int(row["tokens"])
                for row in rows
                if row["tokenizer"] == tokenizer_name and row["lang"] == language
            ]
            averages[tokenizer_name, language] = sum(values) / len(values)

    print("Average tokens per sentence:")
    print("tokenizer  " + "  ".join(f"{language:>8}" for language in LANGUAGES))
    for tokenizer_name in dict.fromkeys(str(row["tokenizer"]) for row in rows):
        values = "  ".join(
            f"{averages[tokenizer_name, language]:8.2f}" for language in LANGUAGES
        )
        print(f"{tokenizer_name:10}{values}")

    print("\nRatio vs English (English should be 1.00):")
    print("tokenizer  " + "  ".join(f"{language:>8}" for language in LANGUAGES))
    for tokenizer_name in dict.fromkeys(str(row["tokenizer"]) for row in rows):
        english_average = averages[tokenizer_name, "en"]
        values = "  ".join(
            f"{averages[tokenizer_name, language] / english_average:8.2f}"
            for language in LANGUAGES
        )
        print(f"{tokenizer_name:10}{values}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "audit/flores200_dataset/devtest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/audit/token_counts.csv",
    )
    parser.add_argument(
        "--pricing",
        type=Path,
        default=ROOT / "audit/pricing.json",
    )
    parser.add_argument("--expected-count", type=int, default=EXPECTED_SENTENCES)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only load Hugging Face tokenizer files already cached locally.",
    )
    args = parser.parse_args()

    sentences = load_sentences(args.dataset_dir, args.expected_count)
    tokenizers = build_tokenizers(offline=args.offline)
    usd_inr, price_entries = load_pricing(args.pricing)
    prices = {name: details for name, details in price_entries.items()}
    rows = count_tokens(sentences, tokenizers, usd_inr, prices)
    write_csv(rows, args.output)
    print_summaries(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
