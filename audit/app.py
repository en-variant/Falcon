"""Interactive, local-first Token Tax demo."""

from __future__ import annotations

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from audit.count_tokens import build_tokenizers, load_pricing
from transform.romanize import romanize_text


ROOT = Path(__file__).resolve().parents[1]
TOKEN_COUNTS_PATH = ROOT / "data/audit/token_counts.csv"
RESULTS_PATH = ROOT / "data/eval/results.csv"
PRICING_PATH = ROOT / "audit/pricing.json"
LANGUAGE_NAMES = {
    "kn": "Kannada",
    "hi": "Hindi",
    "ta": "Tamil",
    "en": "English",
}
LANGUAGE_SAMPLES = {
    "kn": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
    "hi": "नमस्ते, आप कैसे हैं?",
    "ta": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
}
STRATEGIES = ("native", "romanized", "pivot")
LOGGER = logging.getLogger(__name__)
_pivot_translator = None
_pivot_translator_lock = Lock()


@st.cache_resource(show_spinner=False)
def get_tokenizers() -> dict[str, Callable[[str], int]]:
    return build_tokenizers(offline=True)


@st.cache_resource(show_spinner=False)
def get_translation_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="nllb-translation")


def submit_translation(language: str, text: str) -> Future[tuple[str, str]]:
    return get_translation_executor().submit(translate_in_background, text, language)


def translate_in_background(text: str, language: str) -> tuple[str, str]:
    global _pivot_translator
    with _pivot_translator_lock:
        if _pivot_translator is None:
            from transform.pivot import PivotTranslator
            import torch

            torch.set_num_threads(min(torch.get_num_threads(), 4))
            _pivot_translator = PivotTranslator()
        translator = _pivot_translator
    return translator.translate(text, language), translator.device


@st.cache_data(show_spinner=False)
def load_token_counts() -> pd.DataFrame | None:
    if not TOKEN_COUNTS_PATH.is_file():
        return None
    data = pd.read_csv(TOKEN_COUNTS_PATH)
    required = {"id", "lang", "tokenizer", "tokens", "cost_inr"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{TOKEN_COUNTS_PATH} is missing columns: {sorted(missing)}")
    return data


@st.cache_data(show_spinner=False)
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


def count_text(text: str, tokenizers: dict[str, Callable[[str], int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Tokenizer": name, "Tokens": counter(text)}
            for name, counter in tokenizers.items()
        ]
    )


def render_conversion_comparison(
    source_text: str,
    converted_text: str,
    strategy: str,
    pivot_text: str | None = None,
) -> None:
    tokenizers = get_tokenizers()
    usd_inr, prices, pricing_metadata = load_pricing_data()
    labels = {
        "native": "Native",
        "romanized": "Romanized",
        "pivot": "English pivot",
    }
    representations = {"Native": source_text}
    if strategy != "native":
        converted = pivot_text if strategy == "pivot" else converted_text
        if converted:
            representations[labels[strategy]] = converted
    rows = []
    for representation, text in representations.items():
        for counted in count_text(text, tokenizers).to_dict("records"):
            name = counted["Tokenizer"]
            tokens = counted["Tokens"]
            benchmark = prices.get(name)
            cost = (
                tokens * benchmark[0] * usd_inr / 1_000_000
                if benchmark is not None
                else None
            )
            rows.append(
                {
                    "Representation": representation,
                    "Tokenizer": name,
                    "Tokens": tokens,
                    "Reference cost (INR)": cost,
                }
            )
    comparison = pd.DataFrame(rows)
    chart = px.bar(
        comparison,
        x="Tokenizer",
        y="Tokens",
        color="Representation",
        barmode="group",
        labels={"Tokens": "Tokens in this text"},
        title="Live token count by conversion",
        color_discrete_sequence=["#5b67f1", "#25a58b", "#f3a83b"],
    )
    chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(chart, width="stretch")
    st.dataframe(comparison, width="stretch", hide_index=True)
    st.caption(
        "Cost is a reference estimate, not a billed amount. "
        + pricing_metadata["note"]
    )


def select_language_sample(
    language: str, current_text: str, previous_sample: str
) -> str:
    if not current_text.strip() or current_text == previous_sample:
        return LANGUAGE_SAMPLES[language]
    return current_text


def update_language_sample() -> None:
    language = st.session_state["converter_language"]
    previous_sample = st.session_state.get("converter_default_text", "")
    current_text = st.session_state.get("converter_source", "")
    st.session_state["converter_source"] = select_language_sample(
        language, current_text, previous_sample
    )
    st.session_state["converter_default_text"] = LANGUAGE_SAMPLES[language]


def convert_text(text: str, language: str, strategy: str) -> str:
    if strategy == "native":
        return text
    if strategy == "romanized":
        return romanize_text(text, language, "ISO")
    if strategy == "pivot":
        raise ValueError("Pivot conversion must run through the translation worker.")
    raise ValueError(f"Unsupported conversion strategy: {strategy}")


@st.fragment(run_every="1s")
def render_translation_status(cache_key: tuple[str, str]) -> str | None:
    future: Future[tuple[str, str]] | None = st.session_state.get("translation_future")
    if st.session_state.get("translation_key") != cache_key or future is None:
        failure = st.session_state.get("translation_failure")
        if failure and failure[0] == cache_key:
            st.error(failure[1])
            return None
        st.info("Press **Translate to English** to run the real local translation model.")
        return None

    translated = st.session_state.get("translation_result")
    if translated and translated[0] == cache_key:
        text, device = translated[1]
        st.markdown("#### English pivot")
        st.code(text, language="text")
        st.caption(f"Translated with NLLB on {device}.")
        return text

    if not future.done():
        st.info(
            "Translation is running in the background. The app remains usable "
            "while model files download or the model translates on CPU."
        )
        return None

    try:
        result = future.result()
    except Exception as error:
        LOGGER.exception("NLLB translation failed")
        message = (
            "NLLB could not load or translate this text. Check the network and "
            f"model installation, then retry. Details: {error}"
        )
        st.session_state["translation_failure"] = (cache_key, message)
        st.session_state.pop("translation_future", None)
        st.error(message)
        return None

    st.session_state["translation_result"] = (cache_key, result)
    st.session_state.pop("translation_failure", None)
    text, device = result
    st.markdown("#### English pivot")
    st.code(text, language="text")
    st.caption(f"Translated with NLLB on {device}.")
    return text


def render_overview(token_counts: pd.DataFrame | None) -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">AI RESEARCH · THE TOKEN TAX</div>'
        '<h1>Same meaning. Different token bill.</h1>'
        "<p>Measure real tokenizer costs, then compare how your text changes "
        "when it is romanized or translated.</p></div>",
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    a.metric("Aligned FLORES sentences", "1,012 × 4")
    b.metric("Measured token rows", f"{len(token_counts):,}" if token_counts is not None else "Not found")
    if token_counts is not None:
        cl100k = token_counts[token_counts["tokenizer"] == "cl100k"]
        means_by_language = cl100k.groupby("lang")["tokens"].mean()
        ratio = means_by_language["kn"] / means_by_language["en"]
        c.metric(
            "Kannada tax · cl100k",
            f"{ratio:.2f}×",
            help="Measured ratio against English on FLORES-200 devtest.",
        )
    else:
        c.metric("Kannada tax · cl100k", "Unavailable")
    if token_counts is None:
        st.warning("Run `python -m audit.count_tokens` to generate the measured tokenizer data.")
        return

    means = (
        token_counts.groupby(["tokenizer", "lang"], as_index=False)
        .agg(tokens=("tokens", "mean"))
    )
    means["language"] = means["lang"].map(LANGUAGE_NAMES)
    chart = px.bar(
        means,
        x="language",
        y="tokens",
        color="tokenizer",
        barmode="group",
        category_orders={"language": ["English", "Kannada", "Hindi", "Tamil"]},
        labels={"language": "Language", "tokens": "Average tokens / sentence", "tokenizer": "Tokenizer"},
        title="Measured average tokens per aligned sentence",
        color_discrete_sequence=["#5b67f1", "#25a58b", "#f3a83b", "#dc5c7a"],
    )
    chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        margin=dict(l=12, r=12, t=55, b=12),
    )
    st.plotly_chart(chart, width="stretch")
    st.caption(
        "Source: official FLORES-200 devtest. The 12× booklet claim is not "
        "reproduced for cl100k here; ratios vary substantially by tokenizer."
    )


def render_converter() -> None:
    st.markdown("## Live conversion studio")
    st.caption(
        "Choose whether to keep the original script, transliterate it into "
        "Latin letters, or translate its meaning into English."
    )
    if "converter_language" not in st.session_state:
        st.session_state["converter_language"] = "kn"
    if "converter_default_text" not in st.session_state:
        st.session_state["converter_default_text"] = LANGUAGE_SAMPLES["kn"]
    if "converter_source" not in st.session_state:
        st.session_state["converter_source"] = LANGUAGE_SAMPLES["kn"]

    left, right = st.columns([1, 1], gap="large")
    with left:
        language = st.selectbox(
            "Source language",
            ("kn", "hi", "ta"),
            format_func=lambda code: LANGUAGE_NAMES[code],
            key="converter_language",
            on_change=update_language_sample,
        )
        strategy = st.selectbox(
            "Conversion mode",
            STRATEGIES,
            format_func=lambda value: {
                "native": "Native — keep the same text",
                "romanized": "Romanized — change script, keep the wording",
                "pivot": "English pivot — translate the meaning",
            }[value],
            key="converter_strategy",
        )
        source_text = st.text_area(
            "Type or paste your text",
            key="converter_source",
            height=180,
            placeholder=LANGUAGE_SAMPLES[language],
            help=(
                f"Enter {LANGUAGE_NAMES[language]} text. Switching language "
                "also changes the untouched starter example."
            ),
        )

    output = ""
    if source_text.strip():
        try:
            if strategy != "pivot":
                output = convert_text(source_text, language, strategy)
        except (TypeError, ValueError, ImportError) as error:
            st.error(f"Conversion failed: {error}")

    with right:
        output_title = {
            "native": "Native output — unchanged",
            "romanized": "Romanized output — same wording",
            "pivot": "English pivot — meaning translated",
        }[strategy]
        st.markdown(f"#### {output_title}")
        if strategy != "pivot":
            st.code(output or "Conversion output will appear here.", language=None)

    if not source_text.strip():
        st.info("Enter text to see the selected conversion.")
        return

    cache_key = (language, source_text)
    pivot_text = None
    if strategy == "pivot":
        st.caption(
            "English pivot translates meaning using "
            "facebook/nllb-200-distilled-600M. First use downloads model "
            "weights (about 2.4 GB); CPU translation runs in the background "
            "and can take a while."
        )
        pending = st.session_state.get("translation_future")
        failed_key = st.session_state.get("translation_failure")
        translation_failed = failed_key and failed_key[0] == cache_key
        translation_matches = (
            st.session_state.get("translation_key") == cache_key
            and pending is not None
        )
        if not translation_matches and not translation_failed:
            st.session_state["translation_key"] = cache_key
            st.session_state["translation_result"] = None
            st.session_state.pop("translation_failure", None)
            st.session_state["translation_future"] = submit_translation(
                language, source_text
            )
        pivot_text = render_translation_status(cache_key)
        if pivot_text:
            with right:
                st.markdown(f"#### {output_title}")
                st.code(pivot_text, language="text")
    else:
        st.session_state.pop("translation_key", None)
        st.session_state.pop("translation_result", None)
        st.session_state.pop("translation_future", None)
        st.session_state.pop("translation_failure", None)

    st.markdown("#### Compare token cost")
    try:
        render_conversion_comparison(
            source_text,
            output if strategy == "romanized" else source_text,
            strategy,
            pivot_text,
        )
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not count tokens with locally cached tokenizers: {error}")


def render_token_comparison(token_counts: pd.DataFrame | None) -> None:
    st.markdown("## Token inspector")
    if token_counts is None:
        st.warning("Measured audit CSV is unavailable.")
        return
    language = st.selectbox(
        "Compare language",
        ("kn", "hi", "ta"),
        format_func=lambda code: LANGUAGE_NAMES[code],
        key="measure_language",
    )
    means = (
        token_counts[token_counts["lang"].isin(("en", language))]
        .groupby(["tokenizer", "lang"], as_index=False)
        .agg(tokens=("tokens", "mean"))
    )
    pivoted = means.pivot(index="tokenizer", columns="lang", values="tokens")
    pivoted["Indic / English"] = pivoted[language] / pivoted["en"]
    a, b = st.columns([1, 1.5])
    with a:
        st.dataframe(
            pivoted.rename(
                columns={"en": "English avg", language: f"{LANGUAGE_NAMES[language]} avg"}
            ).round(2),
            width="stretch",
        )
    with b:
        chart = px.bar(
            means,
            x="tokenizer",
            y="tokens",
            color="lang",
            barmode="group",
            labels={"tokenizer": "Tokenizer", "tokens": "Average tokens / sentence", "lang": "Language"},
            title=f"{LANGUAGE_NAMES[language]} vs English",
            color_discrete_map={"en": "#25a58b", language: "#5b67f1"},
        )
        chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(chart, width="stretch")


def render_quality(results: pd.DataFrame | None) -> None:
    st.markdown("## Strategy quality — measured evaluation")
    if results is None:
        st.info(
            "No real Member C results.csv is present. Accuracy comparisons are "
            "intentionally hidden until measured Belebele results are supplied."
        )
        return
    language = st.selectbox(
        "Evaluation language",
        ("kn", "hi", "ta", "en"),
        format_func=lambda code: LANGUAGE_NAMES[code],
        key="quality_language",
    )
    rows = results[results["lang"] == language].copy()
    if rows.empty:
        st.info(f"No measured results for {LANGUAGE_NAMES[language]}.")
        return
    rows["correct"] = pd.to_numeric(rows["correct"], errors="raise")
    rows["tokens"] = pd.to_numeric(rows["tokens"], errors="raise")
    summary = rows.groupby("strategy", as_index=False).agg(
        accuracy=("correct", "mean"),
        mean_tokens=("tokens", "mean"),
        questions=("id", "nunique"),
    )
    chart = px.scatter(
        summary,
        x="mean_tokens",
        y="accuracy",
        color="strategy",
        size="questions",
        text="strategy",
        labels={"mean_tokens": "Average input tokens", "accuracy": "Accuracy", "questions": "Questions"},
        title=f"Measured accuracy vs token cost · {LANGUAGE_NAMES[language]}",
    )
    chart.update_yaxes(tickformat=".0%")
    chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(chart, width="stretch")
    st.dataframe(summary, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="The Token Tax",
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #0b1020; color: #eef1ff; }
        [data-testid="stSidebar"] { background: #11182a; }
        .hero { padding: 2rem 2.2rem; border-radius: 20px; margin: .4rem 0 1.2rem;
                background: linear-gradient(115deg,#18254b,#302562 58%,#17534d); }
        .hero h1 { font-size: 2.5rem; margin: .35rem 0; color: white; }
        .hero p { color: #d3dcf6; font-size: 1.05rem; margin-bottom: 0; }
        .eyebrow { color: #b7c3ff; font-size: .75rem; letter-spacing: .16em; font-weight: 700; }
        div[data-testid="stMetric"] { background: #151e33; border: 1px solid #293655;
                padding: 1rem; border-radius: 14px; }
        div[data-testid="stTabs"] button { font-weight: 650; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    try:
        token_counts = load_token_counts()
        results = load_results()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load measured project data: {error}")
        token_counts, results = None, None

    with st.sidebar:
        st.markdown("### THE TOKEN TAX")
        st.caption("Track 03 · AI Research")
        st.markdown("---")
        st.markdown("**Real data only**")
        st.caption("FLORES-200 · 1,012 aligned lines · 4 languages")
        st.caption("Romanization updates as you type. Translation is an explicit model action.")
        st.markdown("---")
        st.caption("Open-weight NLLB translation may be slow on CPU.")

    overview, convert, explore, quality = st.tabs(
        ["Overview", "Convert text", "Token data", "Quality results"]
    )
    with overview:
        render_overview(token_counts)
    with convert:
        render_converter()
    with explore:
        render_token_comparison(token_counts)
    with quality:
        render_quality(results)
    st.caption(
        "OpenAI prices shown in the audit are reference estimates, not billed "
        "KodeKloud usage. Dataset provenance: FLORES-200, CC-BY-SA 4.0."
    )


if __name__ == "__main__":
    main()
