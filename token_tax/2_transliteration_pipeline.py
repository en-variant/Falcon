"""Measure script transliteration and token counts without treating it as translation."""
import json
from pathlib import Path

import tiktoken

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError as exc:
    raise RuntimeError(
        "Install the required indic-transliteration package to run this component."
    ) from exc


ROOT = Path(__file__).resolve().parent
QUESTION_FILE = ROOT / "indian_language_questions.json"
TOKENIZER_NAME = "cl100k_base"
SCRIPT_MAP = {
    "Hindi": sanscript.DEVANAGARI,
    "Tamil": sanscript.TAMIL,
    "Kannada": sanscript.KANNADA,
    "Telugu": sanscript.TELUGU,
    "Bengali": sanscript.BENGALI,
    "Malayalam": sanscript.MALAYALAM,
    "Gujarati": sanscript.GUJARATI,
    "Punjabi": sanscript.GURMUKHI,
    "Marathi": sanscript.DEVANAGARI,
    "Odia": sanscript.ORIYA,
}
UNSUPPORTED_LANGUAGES = {
    "Urdu": "The installed library has no Urdu/Arabic-script mapping; no transliteration was attempted."
}


def to_latin(text: str, language: str) -> str:
    """Transliterate a supported Indic script to ITRANS; never return a placeholder."""
    if language not in SCRIPT_MAP:
        reason = UNSUPPORTED_LANGUAGES.get(language, "Unsupported language or script.")
        raise ValueError(reason)
    return transliterate(text, SCRIPT_MAP[language], sanscript.ITRANS)


def from_latin(text: str, language: str) -> str:
    if language not in SCRIPT_MAP:
        reason = UNSUPPORTED_LANGUAGES.get(language, "Unsupported language or script.")
        raise ValueError(reason)
    return transliterate(text, sanscript.ITRANS, SCRIPT_MAP[language])


def character_similarity(original: str, restored: str) -> float:
    """Return normalized code-point edit similarity; this is not semantic similarity."""
    if original == restored:
        return 100.0
    if not original or not restored:
        return 0.0
    previous = list(range(len(restored) + 1))
    for row, left_char in enumerate(original, start=1):
        current = [row]
        for column, right_char in enumerate(restored, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return round(
        (1 - previous[-1] / max(len(original), len(restored))) * 100,
        2,
    )


def _measure_language(language: str, questions: list[str], encoder) -> dict:
    if language not in SCRIPT_MAP:
        return {
            "status": "unsupported",
            "reason": UNSUPPORTED_LANGUAGES.get(language, "Unsupported language or script."),
            "sample_count": len(questions),
            "samples": [],
            "savings_percent": None,
        }

    samples = []
    original_total = 0
    latin_total = 0
    for index, original in enumerate(questions, start=1):
        latin = to_latin(original, language)
        restored = from_latin(latin, language)
        original_tokens = len(encoder.encode(original))
        latin_tokens = len(encoder.encode(latin))
        original_total += original_tokens
        latin_total += latin_tokens
        samples.append({
            "id": f"q{index:02}",
            "original": original,
            "original_tokens": original_tokens,
            "transliterated": latin,
            "transliterated_tokens": latin_tokens,
            "savings_percent": round((1 - latin_tokens / original_tokens) * 100, 2)
            if original_tokens
            else 0,
            "round_trip": restored,
            "round_trip_exact": restored == original,
            "round_trip_character_similarity_percent": character_similarity(original, restored),
        })

    return {
        "status": "measured",
        "operation": "Indic-script to ITRANS transliteration (not translation)",
        "tokenizer": TOKENIZER_NAME,
        "sample_count": len(samples),
        "original_tokens": original_total,
        "transliterated_tokens": latin_total,
        "savings_percent": round((1 - latin_total / original_total) * 100, 2)
        if original_total
        else 0,
        "exact_round_trips": sum(sample["round_trip_exact"] for sample in samples),
        "mean_round_trip_character_similarity_percent": round(
            sum(sample["round_trip_character_similarity_percent"] for sample in samples)
            / len(samples),
            2,
        ) if samples else 0,
        "samples": samples,
    }


def run_pipeline() -> dict:
    dataset = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)
    return {
        language: _measure_language(language, questions, encoder)
        for language, questions in dataset["languages"].items()
    }


def main() -> None:
    results = run_pipeline()
    print("\nTRANSLITERATION — measured cl100k counts; this is not translation")
    for language, result in results.items():
        if result["status"] == "unsupported":
            print(f"{language}: unsupported — {result['reason']}")
            continue
        print(
            f"{language}: {result['original_tokens']} -> "
            f"{result['transliterated_tokens']} tokens; "
            f"{result['savings_percent']}% measured; "
            f"{result['exact_round_trips']}/{result['sample_count']} exact round trips; "
            f"mean character similarity "
            f"{result['mean_round_trip_character_similarity_percent']}% "
            "(not semantic accuracy)"
        )
    output = ROOT / "pipeline_results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
