"""Offline Streamlit demo for the measured token-tax audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from audit.count_tokens import LANGUAGES, build_tokenizers, load_pricing
from router.features import SUPPORTED_LANGUAGES
from router.transforms import apply_strategy


ROOT = Path(__file__).resolve().parents[1]
TOKEN_COUNTS_PATH = ROOT / "data/audit/token_counts.csv"
RESULTS_PATH = ROOT / "data/eval/results.csv"
PREDICTIONS_PATH = ROOT / "data/router/predictions.csv"
FAILURE_CASES_PATH = ROOT / "audit/failure_cases.csv"
PRICING_PATH = ROOT / "audit/pricing.json"
STRATEGIES = ("native", "romanized", "pivot")


@st.cache_resource(show_spinner=False)
def get_tokenizers():
    return build_tokenizers(offline=True)


def load_token_counts() -> pd.DataFrame | None:
    if not TOKEN_COUNTS_PATH.is_file():
        return None
    data = pd.read_csv(TOKEN_COUNTS_PATH)
    required = {"id", "lang", "tokenizer", "tokens", "cost_inr"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{TOKEN_COUNTS_PATH} is missing columns: {sorted(missing)}")
    return data


def load_results() -> pd.DataFrame | None:
    if not RESULTS_PATH.is_file():
        return None
    results = pd.read_csv(RESULTS_PATH)
    required = {"id", "lang", "strategy", "correct", "tokens"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"{RESULTS_PATH} is missing columns: {sorted(missing)}")
    return results


def load_pricing_data() -> tuple[float, dict[str, tuple[float, str]], dict]:
    with PRICING_PATH.open(encoding="utf-8") as source:
        metadata = json.load(source)
    exchange_rate, benchmarks = load_pricing(PRICING_PATH)
    return exchange_rate, benchmarks, metadata


def render_inspector() -> None:
    st.subheader("1. Live Tokenizer Inspector")
    left, right = st.columns(2)
    with left:
        language = st.selectbox(
            "Indic language",
            options=("kn", "hi", "ta"),
            format_func=lambda value: {"kn": "Kannada", "hi": "Hindi", "ta": "Tamil"}[
                value
            ],
        )
        indic_text = st.text_area(
            "Paste Indic text",
            placeholder="Paste text to count with the local tokenizers.",
            height=130,
        )
    with right:
        english_text = st.text_area(
            "Same meaning in English (optional)",
            placeholder="Paste the aligned English text for comparison.",
            height=130,
        )

    if not indic_text.strip():
        st.info("Enter text to see actual local tokenizer counts.")
        return
    try:
        tokenizers = get_tokenizers()
        usd_inr, prices, pricing_metadata = load_pricing_data()
    except Exception as error:
        st.error(f"Could not load local tokenizers or pricing data: {error}")
        return

    texts = {language: indic_text}
    if english_text.strip():
        texts["en"] = english_text
    rows = []
    for tokenizer_name, counter in tokenizers.items():
        for text_language, text in texts.items():
            token_count = counter(text)
            benchmark = prices.get(tokenizer_name)
            cost = (
                token_count * benchmark[0] * usd_inr / 1_000_000
                if benchmark is not None
                else None
            )
            rows.append(
                {
                    "Language": text_language,
                    "Tokenizer": tokenizer_name,
                    "Tokens": token_count,
                    "Estimated cost (INR)": cost,
                }
            )
    counts = pd.DataFrame(rows)
    st.dataframe(counts, width="stretch", hide_index=True)

    if english_text.strip():
        ratios = counts.pivot(
            index="Tokenizer", columns="Language", values="Tokens"
        )
        ratios["Indic / English"] = ratios[language] / ratios["en"].replace(0, pd.NA)
        st.write("**Indic-to-English token ratio**")
        st.dataframe(
            ratios[["Indic / English"]].round(2),
            width="stretch",
        )
    st.caption(
        "API costs are reference estimates from pricing.json. "
        + pricing_metadata["note"]
    )


def render_strategy_simulator(results: pd.DataFrame | None) -> None:
    st.subheader("2. Strategy Simulator")
    language = st.selectbox(
        "Text language",
        options=SUPPORTED_LANGUAGES,
        key="simulator_language",
    )
    strategy = st.selectbox("Strategy", options=STRATEGIES, key="simulator_strategy")
    original = st.text_area(
        "Input text",
        key="simulator_text",
        placeholder="Enter real text to inspect a configured transformation.",
    )
    if original.strip():
        try:
            transformed = apply_strategy(original, strategy, language)
            st.markdown("**Result**")
            st.code(transformed)
        except Exception as error:
            st.warning(f"Transformation unavailable: {error}")
            transformed = None
    else:
        transformed = None

    if results is None:
        st.info("Quality/latency metrics unavailable until Member C's real results.csv exists.")
        return
    matching = results[
        (results["lang"] == language) & (results["strategy"] == strategy)
    ]
    if matching.empty:
        st.info(f"No measured results for {language}/{strategy} yet.")
        return
    metrics = {
        "Accuracy": float(pd.to_numeric(matching["correct"]).mean()),
        "Mean model input tokens": float(pd.to_numeric(matching["tokens"]).mean()),
    }
    if "latency_seconds" in matching.columns:
        metrics["Mean latency (seconds)"] = float(
            pd.to_numeric(matching["latency_seconds"], errors="coerce").mean()
        )
    elif "latency_ms" in matching.columns:
        metrics["Mean latency (milliseconds)"] = float(
            pd.to_numeric(matching["latency_ms"], errors="coerce").mean()
        )
    st.json(metrics)
    if transformed is not None:
        st.caption(
            "Token/quality values above are measured evaluation aggregates, not "
            "predictions for this newly entered text."
        )
    if "latency_seconds" not in matching and "latency_ms" not in matching:
        st.caption("Latency was not recorded in results.csv.")


def render_pareto_chart(results: pd.DataFrame | None) -> None:
    st.subheader("3. Measured Accuracy vs Input Tokens")
    if results is None:
        st.info("Chart unavailable until Member C's real results.csv exists.")
        return
    language = st.selectbox(
        "Chart language",
        options=SUPPORTED_LANGUAGES,
        key="chart_language",
    )
    filtered = results[results["lang"] == language].copy()
    if filtered.empty:
        st.info(f"No measured results for {language} yet.")
        return
    filtered["correct"] = pd.to_numeric(filtered["correct"], errors="raise")
    filtered["tokens"] = pd.to_numeric(filtered["tokens"], errors="raise")
    points = (
        filtered.groupby("strategy", as_index=False)
        .agg(accuracy=("correct", "mean"), mean_tokens=("tokens", "mean"))
    )

    if PREDICTIONS_PATH.is_file():
        predictions = pd.read_csv(
            PREDICTIONS_PATH, dtype={"id": "string", "chosen_strategy": "string"}
        )
        if {"id", "chosen_strategy"}.issubset(predictions.columns):
            language_ids = set(filtered["id"].astype(str))
            predictions = predictions[
                predictions["id"].astype(str).isin(language_ids)
            ]
            chosen = predictions.merge(
                filtered,
                left_on=["id", "chosen_strategy"],
                right_on=["id", "strategy"],
                how="inner",
            )
            if not chosen.empty:
                points.loc[len(points)] = {
                    "strategy": "router (out-of-fold)",
                    "accuracy": float(chosen["correct"].mean()),
                    "mean_tokens": float(chosen["tokens"].mean()),
                }
    if points.empty:
        st.info("No strategy aggregates available for this language.")
        return
    chart = px.scatter(
        points,
        x="mean_tokens",
        y="accuracy",
        color="strategy",
        hover_data={"mean_tokens": ":.2f", "accuracy": ":.3f"},
        labels={
            "mean_tokens": "Average input tokens",
            "accuracy": "Measured accuracy",
            "strategy": "Strategy",
        },
        title=f"{language.upper()}: quality and input-token tradeoff",
    )
    chart.update_yaxes(tickformat=".0%")
    st.plotly_chart(chart, width="stretch")
    st.caption(
        "This uses measured Member C results. Router point is calculated only "
        "from saved out-of-fold predictions that match these results."
    )


def render_failure_cases() -> None:
    st.subheader("4. Measured Failure Cases")
    if not FAILURE_CASES_PATH.is_file():
        st.info(
            "No real failure-case file is available. Add manually verified examples "
            "to audit/failure_cases.csv; the app will not invent examples."
        )
        return
    cases = pd.read_csv(FAILURE_CASES_PATH)
    required = {"lang", "strategy", "input", "output", "failure_type"}
    missing = required - set(cases.columns)
    if missing:
        st.error(f"failure_cases.csv is missing columns: {sorted(missing)}")
        return
    if cases.empty:
        st.info("The failure-case file exists but has no verified cases yet.")
        return
    if st.button("Show a verified failure case"):
        index = int(st.session_state.get("failure_case_index", -1)) + 1
        st.session_state["failure_case_index"] = index % len(cases)
    index = st.session_state.get("failure_case_index")
    if index is None:
        st.info(f"{len(cases)} verified examples loaded. Click the button to inspect one.")
        return
    row = cases.iloc[index]
    st.markdown(
        f"**{row['lang']} · {row['strategy']} · {row['failure_type']}**"
    )
    st.write("**Input**")
    st.write(row["input"])
    st.write("**Transformed output**")
    st.write(row["output"])
    if "notes" in cases.columns and pd.notna(row["notes"]):
        st.caption(str(row["notes"]))


def main() -> None:
    st.set_page_config(page_title="The Token Tax", page_icon="🧮", layout="wide")
    st.title("The Token Tax")
    st.caption(
        "Measured tokenizer comparison for English, Kannada, Hindi, and Tamil. "
        "Panels with missing team data stay unavailable rather than showing mock results."
    )
    st.info(
        "FLORES-200 is CC-BY-SA 4.0. Dataset provenance and price caveats are in audit/README.md."
    )

    try:
        token_counts = load_token_counts()
        results = load_results()
    except Exception as error:
        st.error(f"Could not load local audit data: {error}")
        token_counts = None
        results = None

    if token_counts is None:
        st.warning(
            "Token audit CSV not found. Run `python -m audit.count_tokens` after "
            "the FLORES files and tokenizer assets are available."
        )
    else:
        st.caption(f"Loaded {len(token_counts):,} measured token-count rows.")

    render_inspector()
    st.divider()
    render_strategy_simulator(results)
    st.divider()
    render_pareto_chart(results)
    st.divider()
    render_failure_cases()


if __name__ == "__main__":
    main()
