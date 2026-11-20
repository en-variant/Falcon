"""Count Gemini input tokens and estimate LLM input costs for sample texts.

The default texts are examples only. Counts are obtained from Google's Gemini
countTokens API when run with credentials; no sample token counts are fabricated.
This module estimates LLM input costs only. It does not translate text or include
translation-service costs.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Protocol

import matplotlib.pyplot as plt
import pandas as pd

if __package__:
    from .cost_calculator import calculate_input_cost
else:
    from cost_calculator import calculate_input_cost


MODEL_ID = "gemini-3.8-flash"
API_KEY_ENVIRONMENT_VARIABLE = "GEMINI_API_KEY"
PRICE_ENVIRONMENT_VARIABLE = "GEMINI_INPUT_PRICE_PER_MILLION_TOKENS_USD"
REPRESENTATIONS = ("native_kannada", "romanized_kannada", "english")
DATASET_TEXT_COLUMNS = (
    "native_kannada",
    "romanized_kannada",
    "english",
)
SAMPLE_TEXTS = {
    "native_kannada": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
    "romanized_kannada": "Namaskara, neevu hegiddiri?",
    "english": "Hello, how are you?",
}

TOKENIZER_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = TOKENIZER_DIRECTORY / "gemini_3_8_flash_cost_comparison.csv"
DEFAULT_GRAPH_PATH = (
    TOKENIZER_DIRECTORY / "graphs" / "gemini_3_8_flash_input_cost_comparison.png"
)
DEFAULT_DATASET_CSV_PATH = TOKENIZER_DIRECTORY / "test_dataset.csv"
DEFAULT_DATASET_OUTPUT_PATH = (
    TOKENIZER_DIRECTORY / "gemini_dataset_cost_comparison.csv"
)


class CountTokensResponse(Protocol):
    """The count field returned by Google's countTokens API."""

    total_tokens: Optional[int]


class GeminiModelsClient(Protocol):
    """The Gemini SDK model methods used by this module."""

    def count_tokens(self, *, model: str, contents: str) -> CountTokensResponse:
        """Return the token count for the provided contents."""


class GeminiClient(Protocol):
    """The subset of the Google GenAI client used for token counting."""

    models: GeminiModelsClient


class TokenCounter(Protocol):
    """A token counter interface that also allows isolated test clients."""

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in text."""


class GeminiTokenCounter:
    """Use Google's official Gemini countTokens endpoint for exact counts."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        client: Optional[GeminiClient] = None,
    ) -> None:
        self.model_id = model_id
        try:
            from google import genai
            from google.genai import errors
        except ImportError as error:
            raise ImportError(
                "Install Gemini support with "
                "'python -m pip install -r tokenizer\\requirements.txt'."
            ) from error

        self._api_error_type = errors.APIError
        if client is not None:
            self._client = client
            return

        api_key = os.environ.get(API_KEY_ENVIRONMENT_VARIABLE)
        if not api_key:
            raise RuntimeError(
                f"Set {API_KEY_ENVIRONMENT_VARIABLE} in your environment before "
                "using the Gemini API. The key is never written to the report."
            )

        self._client = genai.Client(api_key=api_key)

    def count_tokens(self, text: str, context: Optional[str] = None) -> int:
        """Count input tokens via Gemini's official API."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        try:
            response = self._client.models.count_tokens(
                model=self.model_id,
                contents=text,
            )
        except self._api_error_type as error:
            location = f" for {context}" if context else ""
            raise RuntimeError(
                f"Gemini countTokens API failed{location}: {error}"
            ) from error

        token_count = response.total_tokens
        if isinstance(token_count, bool) or not isinstance(token_count, int):
            raise RuntimeError(
                "Gemini countTokens response did not contain a valid total_tokens."
            )
        if token_count < 0:
            raise RuntimeError(
                "Gemini countTokens response returned a negative total_tokens."
            )
        return token_count


def get_input_price(configured_price: Optional[float] = None) -> float:
    """Read the USD price per million tokens from an argument or environment."""
    if configured_price is None:
        raw_price = os.environ.get(PRICE_ENVIRONMENT_VARIABLE)
        if raw_price is None:
            raise RuntimeError(
                f"Provide --input-price-per-million-tokens-usd or set "
                f"{PRICE_ENVIRONMENT_VARIABLE}. Use the rate for your Gemini "
                "billing tier; this module does not assume a price."
            )
        try:
            configured_price = float(raw_price)
        except ValueError as error:
            raise ValueError(
                f"{PRICE_ENVIRONMENT_VARIABLE} must be a numeric USD price."
            ) from error

    if not math.isfinite(configured_price) or configured_price < 0:
        raise ValueError("The configured input price must be finite and non-negative.")
    return configured_price


def build_cost_report(
    texts: Dict[str, str],
    input_price_per_million_tokens_usd: float,
    token_counter: TokenCounter,
    input_set_type: str,
) -> pd.DataFrame:
    """Count each representation and estimate its LLM input cost.

    ``input_set_type`` labels the text set as ``sample`` or ``user_provided``.
    The measurements themselves always come from the supplied token counter.
    """
    missing_representations = [
        name for name in REPRESENTATIONS if name not in texts
    ]
    if missing_representations:
        missing = ", ".join(missing_representations)
        raise ValueError(f"Missing text representations: {missing}")
    if input_set_type not in ("sample", "user_provided"):
        raise ValueError("input_set_type must be 'sample' or 'user_provided'")

    # Reuse the project's input-cost calculation and its price validation.
    calculate_input_cost(0, input_price_per_million_tokens_usd)

    rows = []
    for representation in REPRESENTATIONS:
        text = texts[representation]
        if not isinstance(text, str):
            raise TypeError(f"{representation} text must be a string")
        token_count = token_counter.count_tokens(text)
        estimated_cost = calculate_input_cost(
            token_count, input_price_per_million_tokens_usd
        )
        rows.append(
            {
                "representation": representation,
                "model_id": MODEL_ID,
                "token_count": token_count,
                "token_count_source": "Gemini countTokens API",
                "measurement_status": "actual_gemini_api_measurement",
                "input_set_type": input_set_type,
                "cost_type": "LLM input only",
                "input_price_per_million_tokens_usd": (
                    input_price_per_million_tokens_usd
                ),
                "estimated_llm_input_cost_usd": estimated_cost,
            }
        )

    return pd.DataFrame(rows)


def save_cost_graph(report: pd.DataFrame, output_path: Path) -> Path:
    """Plot estimated LLM input costs by representation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_set_type = str(report["input_set_type"].iloc[0])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        report["representation"],
        report["estimated_llm_input_cost_usd"],
        color="#4472C4",
    )
    axis.set_title(
        f"{MODEL_ID} Gemini-Measured LLM Input Cost ({input_set_type} text)"
    )
    axis.set_xlabel("Text representation")
    axis.set_ylabel("Estimated LLM input cost (USD)")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def generate_cost_comparison(
    texts: Dict[str, str],
    input_price_per_million_tokens_usd: float,
    input_set_type: str,
    token_counter: Optional[TokenCounter] = None,
    csv_path: Path = DEFAULT_CSV_PATH,
    graph_path: Path = DEFAULT_GRAPH_PATH,
) -> pd.DataFrame:
    """Generate a CSV and graph from Gemini token counts and configured prices."""
    if token_counter is None:
        token_counter = GeminiTokenCounter()

    report = build_cost_report(
        texts=texts,
        input_price_per_million_tokens_usd=input_price_per_million_tokens_usd,
        token_counter=token_counter,
        input_set_type=input_set_type,
    )

    csv_path = Path(csv_path)
    graph_path = Path(graph_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(csv_path, index=False)
    save_cost_graph(report, graph_path)
    print(f"Gemini token measurements saved to: {csv_path}")
    print(f"LLM input cost graph saved to: {graph_path}")
    print(
        f"Input set: {input_set_type}. Token counts came from Gemini's "
        f"countTokens API using {MODEL_ID}."
    )
    print("Translation-service costs are separate and are not included.")
    return report


def read_dataset_csv(dataset_csv_path: Path = DEFAULT_DATASET_CSV_PATH) -> pd.DataFrame:
    """Load and validate sentence IDs and the three required text columns."""
    dataset_csv_path = Path(dataset_csv_path)
    text_types = {column: "string" for column in DATASET_TEXT_COLUMNS}
    text_types["id"] = "string"
    dataset = pd.read_csv(
        dataset_csv_path,
        dtype=text_types,
        keep_default_na=False,
    )

    required_columns = {"id", *DATASET_TEXT_COLUMNS}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset CSV is missing required columns: {missing}")
    if dataset.empty:
        raise ValueError(f"Dataset CSV contains no rows: {dataset_csv_path}")
    if dataset["id"].str.strip().eq("").any():
        raise ValueError("Every dataset row must have a non-empty id.")
    return dataset


def build_dataset_cost_report(
    dataset: pd.DataFrame,
    input_price_per_million_tokens_usd: float,
) -> pd.DataFrame:
    """Measure all dataset texts with Gemini and calculate input costs.

    Each representation is sent in a separate countTokens API request. No
    estimated or locally generated token counts are used as substitutes.
    """
    calculate_input_cost(0, input_price_per_million_tokens_usd)
    token_counter = GeminiTokenCounter()
    rows = []

    for _, sentence in dataset.iterrows():
        sentence_id = str(sentence["id"])
        for representation in REPRESENTATIONS:
            text = sentence[representation]
            if not isinstance(text, str):
                raise TypeError(
                    f"Text for sentence id {sentence_id}, "
                    f"representation {representation} must be a string."
                )

            token_count = token_counter.count_tokens(
                text,
                context=(
                    f"sentence id {sentence_id}, "
                    f"representation {representation}"
                ),
            )
            estimated_cost = calculate_input_cost(
                token_count,
                input_price_per_million_tokens_usd,
            )
            rows.append(
                {
                    "sentence_id": sentence_id,
                    "representation": representation,
                    "token_count": token_count,
                    "estimated_input_cost_usd": estimated_cost,
                    "model_id": MODEL_ID,
                    "token_count_source": "Gemini countTokens API",
                    "measurement_status": "actual_gemini_api_measurement",
                    "input_price_per_million_tokens_usd": (
                        input_price_per_million_tokens_usd
                    ),
                }
            )

    return pd.DataFrame(rows)


def generate_dataset_cost_comparison(
    dataset_csv_path: Path = DEFAULT_DATASET_CSV_PATH,
    output_csv_path: Path = DEFAULT_DATASET_OUTPUT_PATH,
    input_price_per_million_tokens_usd: Optional[float] = None,
) -> pd.DataFrame:
    """Measure the dataset and write its Gemini input-cost comparison CSV."""
    input_price = get_input_price(input_price_per_million_tokens_usd)
    dataset = read_dataset_csv(dataset_csv_path)
    report = build_dataset_cost_report(dataset, input_price)

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv_path, index=False)
    print(f"Gemini dataset measurements saved to: {output_csv_path}")
    print(
        f"Measured {len(dataset)} sentence rows across {len(REPRESENTATIONS)} "
        f"representations with {MODEL_ID}."
    )
    return report


def load_texts(input_json_path: Optional[Path]) -> tuple:
    """Load user text from JSON, or return the built-in sample texts."""
    if input_json_path is None:
        return SAMPLE_TEXTS, "sample"

    with Path(input_json_path).open("r", encoding="utf-8") as input_file:
        texts = json.load(input_file)
    if not isinstance(texts, dict):
        raise ValueError("The input JSON must contain an object of text values.")
    return texts, "user_provided"


def main() -> None:
    """Run Gemini token counting and create the cost comparison artifacts."""
    parser = argparse.ArgumentParser(
        description=(
            "Count sample or supplied text with Gemini and estimate LLM "
            "input costs at your configured price."
        )
    )
    parser.add_argument(
        "--input-price-per-million-tokens-usd",
        type=float,
        help=(
            "Your configured USD input price per million tokens. If omitted, "
            f"reads {PRICE_ENVIRONMENT_VARIABLE}."
        ),
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--inputs-json",
        type=Path,
        help=(
            "Optional JSON object with native_kannada, romanized_kannada, "
            "and english string values. Without this, sample text is used."
        ),
    )
    input_group.add_argument(
        "--dataset-csv",
        nargs="?",
        const=DEFAULT_DATASET_CSV_PATH,
        type=Path,
        help=(
            "Measure every row in the dataset CSV. If no path is supplied, "
            f"uses {DEFAULT_DATASET_CSV_PATH}."
        ),
    )
    parser.add_argument(
        "--dataset-output-csv",
        type=Path,
        default=DEFAULT_DATASET_OUTPUT_PATH,
        help=f"Dataset report path (default: {DEFAULT_DATASET_OUTPUT_PATH}).",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    arguments = parser.parse_args()

    input_price = get_input_price(
        arguments.input_price_per_million_tokens_usd
    )
    if arguments.dataset_csv is not None:
        try:
            generate_dataset_cost_comparison(
                dataset_csv_path=arguments.dataset_csv,
                output_csv_path=arguments.dataset_output_csv,
                input_price_per_million_tokens_usd=input_price,
            )
        except (OSError, RuntimeError, ValueError) as error:
            parser.exit(1, f"Error: {error}\n")
        return

    texts, input_set_type = load_texts(arguments.inputs_json)
    generate_cost_comparison(
        texts=texts,
        input_price_per_million_tokens_usd=input_price,
        input_set_type=input_set_type,
        csv_path=arguments.csv,
        graph_path=arguments.graph,
    )


if __name__ == "__main__":
    main()
