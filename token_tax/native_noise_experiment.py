"""Measure conservative native-script cleanup on clean and noise-augmented queries."""
import json
import hashlib
import platform
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent
SOURCE_FILE = ROOT / "indian_language_questions.json"
DATASET_FILE = ROOT / "native_noise_dataset.json"
RESULT_FILE = ROOT / "native_noise_results.json"
SYSTEM_PROMPT = "Answer the user's question accurately and concisely in the language used by the user."
QUALITY_RUBRIC = """Compare two answers to the same user's question. Score each answer from 1 to 5 for factual accuracy, relevance, and completeness. Also decide whether the two question phrasings preserve the same intent. Do not reward either answer for being longer. Return only a JSON object with these keys:
{"intent_preserved": true, "original_answer_quality": 1, "optimized_answer_quality": 1, "notes": "brief reason"}
Use integer quality scores from 1 (poor) to 5 (excellent)."""
TOKENIZERS = {
    "cl100k_base": "cl100k_base",
    "o200k_base": "o200k_base",
    "p50k_base": "p50k_base",
}
CONTEXT_PREFIXES = {
    "Hindi": "संदर्भ के लिए: ",
    "Tamil": "சூழலுக்காக: ",
    "Kannada": "ಸಂದರ್ಭಕ್ಕಾಗಿ: ",
    "Telugu": "సందర్భం కోసం: ",
    "Bengali": "প্রসঙ্গের জন্য: ",
    "Malayalam": "സന്ദർഭത്തിനായി: ",
    "Gujarati": "સંદર્ભ માટે: ",
    "Punjabi": "ਸੰਦਰਭ ਲਈ: ",
    "Marathi": "संदर्भासाठी: ",
    "Odia": "ପ୍ରସଙ୍ଗ ପାଇଁ: ",
    "Urdu": "سیاق کے لیے: ",
}
TECHNIQUES = (
    "unicode_nfc",
    "horizontal_whitespace",
    "outer_whitespace_trim",
    "blank_line_compaction",
    "combined_safe_normalization",
    "repetition_detection_only",
)


def build_dataset() -> dict:
    source_bytes = SOURCE_FILE.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    records = []
    for language, questions in source["languages"].items():
        if language not in CONTEXT_PREFIXES:
            raise ValueError(f"No native-script context prefix is defined for {language}.")
        for index, clean_query in enumerate(questions, start=1):
            base_id = f"{language}-q{index:02}"
            prefix = CONTEXT_PREFIXES[language]
            conditions = (
                ("clean_control", clean_query, []),
                (
                    "whitespace_noise",
                    " \t  " + clean_query.replace(" ", " \t   ") + " \t ",
                    ["repeated horizontal whitespace", "outer whitespace"],
                ),
                (
                    "formatting_noise",
                    "\r\n\r\n   "
                    + clean_query.replace(" ", "\r\n\r\n  \r\n", 1)
                    + "  \r\n\r\n\r\n",
                    [
                        "leading/trailing blank lines",
                        "internal blank-line run",
                        "line indentation",
                        "outer whitespace",
                    ],
                ),
                (
                    "repeated_query",
                    clean_query + "  " + clean_query,
                    ["repeated complete query", "repeated horizontal whitespace"],
                ),
                (
                    "redundant_context",
                    prefix + clean_query + "\n" + prefix + clean_query,
                    ["repeated context line", "repeated query"],
                ),
            )
            for condition, original_query, noise_types in conditions:
                records.append({
                    "id": f"{base_id}-{condition}",
                    "base_id": base_id,
                    "language": language,
                    "condition": condition,
                    "noise_types": noise_types,
                    "clean_control_query": clean_query,
                    "original_query": original_query,
                    "synthetic_augmentation": condition != "clean_control",
                    "source_type": "curated draft; not collected user traffic",
                })
    return {
        "dataset_version": 1,
        "source_dataset": SOURCE_FILE.name,
        "source_dataset_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "description": (
            "110 clean controls plus deterministic formatting/repetition variants. "
            "Variants are synthetic realistic-style perturbations, not observations "
            "from real users; source prompts require native-speaker review."
        ),
        "records": records,
    }


def normalize_unicode_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_horizontal_whitespace(text: str) -> str:
    return re.sub(r"[ \t\f\v]+", " ", text).strip(" \t\f\v")


def trim_outer_whitespace(text: str) -> str:
    return text.strip(" \t\f\v\r\n")


def compact_blank_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)
    return text.strip(" \t\f\v\r\n")


def combined_safe_normalization(text: str) -> str:
    return compact_blank_lines(normalize_horizontal_whitespace(normalize_unicode_nfc(text)))


def detect_repeated_query(text: str) -> bool:
    normalized = normalize_horizontal_whitespace(text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if any(left == right for left, right in zip(lines, lines[1:])):
        return True
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[?؟।.!])\s+", normalized)
        if segment.strip()
    ]
    return any(left == right for left, right in zip(segments, segments[1:]))


def optimize(text: str, technique: str) -> tuple[str, bool]:
    if technique == "unicode_nfc":
        return normalize_unicode_nfc(text), False
    if technique == "horizontal_whitespace":
        return normalize_horizontal_whitespace(text), False
    if technique == "outer_whitespace_trim":
        return trim_outer_whitespace(text), False
    if technique == "blank_line_compaction":
        return compact_blank_lines(text), False
    if technique == "combined_safe_normalization":
        return combined_safe_normalization(text), False
    if technique == "repetition_detection_only":
        return text, detect_repeated_query(text)
    raise ValueError(f"Unknown optimization technique: {technique}")


def _token_counts(original: str, optimized: str, encoders: dict) -> dict:
    counts = {}
    for name, encoder in encoders.items():
        original_count = len(encoder.encode(original, disallowed_special=()))
        optimized_count = len(encoder.encode(optimized, disallowed_special=()))
        saved = original_count - optimized_count
        counts[name] = {
            "original": original_count,
            "optimized": optimized_count,
            "saved_tokens": saved,
            "savings_percent": round(saved / original_count * 100, 3)
            if original_count
            else 0,
        }
    return counts


def _textual_invariant(original: str, optimized: str) -> bool:
    def content_without_whitespace(text: str) -> str:
        content = "".join(char for char in text if not char.isspace())
        return unicodedata.normalize("NFD", content)

    return content_without_whitespace(original) == content_without_whitespace(optimized)


def _new_accumulator() -> dict:
    return {
        "sample_count": 0,
        "changed_text_samples": 0,
        "repetition_candidates_detected": 0,
        "textual_invariant_failures": 0,
        "by_tokenizer": {
            name: {
                "original_tokens": 0,
                "optimized_tokens": 0,
                "saved_tokens": 0,
                "increased_samples": 0,
                "reduced_samples": 0,
                "unchanged_samples": 0,
            }
            for name in TOKENIZERS
        },
    }


def _add_measurement(accumulator: dict, row: dict, technique: str) -> None:
    measurement = row["techniques"][technique]
    accumulator["sample_count"] += 1
    accumulator["changed_text_samples"] += measurement["changed"]
    accumulator["repetition_candidates_detected"] += measurement["candidate_detected"]
    accumulator["textual_invariant_failures"] += not measurement["textual_invariant_passed"]
    for name, counts in measurement["token_counts"].items():
        summary = accumulator["by_tokenizer"][name]
        summary["original_tokens"] += counts["original"]
        summary["optimized_tokens"] += counts["optimized"]
        summary["saved_tokens"] += counts["saved_tokens"]
        if counts["saved_tokens"] > 0:
            summary["reduced_samples"] += 1
        elif counts["saved_tokens"] < 0:
            summary["increased_samples"] += 1
        else:
            summary["unchanged_samples"] += 1


def _finish_accumulator(accumulator: dict) -> dict:
    for counts in accumulator["by_tokenizer"].values():
        counts["savings_percent"] = round(
            counts["saved_tokens"] / counts["original_tokens"] * 100, 3
        ) if counts["original_tokens"] else 0
    return accumulator


def run_experiment() -> dict:
    dataset = build_dataset()
    encoders = {
        name: tiktoken.get_encoding(encoding)
        for name, encoding in TOKENIZERS.items()
    }
    results = []
    aggregate = {
        technique: {
            "overall": _new_accumulator(),
            "by_language": defaultdict(_new_accumulator),
            "by_condition": defaultdict(_new_accumulator),
        }
        for technique in TECHNIQUES
    }

    for record in dataset["records"]:
        technique_results = {}
        for technique in TECHNIQUES:
            optimized, candidate_detected = optimize(record["original_query"], technique)
            technique_results[technique] = {
                "optimized_query": optimized,
                "changed": optimized != record["original_query"],
                "candidate_detected": candidate_detected,
                "candidate_action": (
                    "not_removed_pending_semantic_evaluation"
                    if candidate_detected
                    else "none"
                ),
                "textual_invariant_passed": _textual_invariant(
                    record["original_query"], optimized
                ),
                "token_counts": _token_counts(record["original_query"], optimized, encoders),
                "semantic_evaluation": "not_run",
            }
        row = {
            **record,
            "evaluation_pair_ids": {
                technique: f"{record['id']}:{technique}"
                for technique in TECHNIQUES
            },
            "techniques": technique_results,
        }
        results.append(row)
        for technique in TECHNIQUES:
            _add_measurement(aggregate[technique]["overall"], row, technique)
            _add_measurement(
                aggregate[technique]["by_language"][record["language"]], row, technique
            )
            _add_measurement(
                aggregate[technique]["by_condition"][record["condition"]], row, technique
            )

    summaries = {}
    for technique, values in aggregate.items():
        summaries[technique] = {
            "description": (
                "Detect exact repeated question segments; leave text unchanged because "
                "repetition can express emphasis or intent."
                if technique == "repetition_detection_only"
                else f"Apply {technique.replace('_', ' ')} to the original query only."
            ),
            "overall": _finish_accumulator(values["overall"]),
            "by_language": {
                language: _finish_accumulator(summary)
                for language, summary in sorted(values["by_language"].items())
            },
            "by_condition": {
                condition: _finish_accumulator(summary)
                for condition, summary in sorted(values["by_condition"].items())
            },
        }

    return {
        "experiment": {
            "dataset_version": dataset["dataset_version"],
            "dataset_description": dataset["description"],
            "query_count": len(results),
            "clean_control_count": sum(
                record["condition"] == "clean_control" for record in results
            ),
            "synthetic_variant_count": sum(
                record["synthetic_augmentation"] for record in results
            ),
            "language_count": len({record["language"] for record in results}),
            "tokenizers": TOKENIZERS,
            "runtime": {
                "python_version": platform.python_version(),
                "tiktoken_version": tiktoken.__version__,
                "source_dataset_sha256": dataset["source_dataset_sha256"],
            },
            "semantic_preservation": (
                "Not established by token counts or text normalization. No repeated "
                "query/context is deleted; candidate removal remains gated on evaluation."
            ),
            "external_evaluation": {
                "status": "prepared_not_run",
                "same_model_required": True,
                "model": None,
                "temperature": 0,
                "answer_system_prompt": SYSTEM_PROMPT,
                "quality_rubric": QUALITY_RUBRIC,
                "pairing": (
                    "Each technique result has a stable evaluation_pair_id and retains "
                    "the unchanged original_query."
                ),
                "note": "The dataset is not sent to any external model by this experiment.",
            },
            "techniques": summaries,
        },
        "results": results,
    }


def main() -> dict:
    dataset = build_dataset()
    DATASET_FILE.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = run_experiment()
    RESULT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Native noise experiment: {report['experiment']['query_count']} queries "
        f"({report['experiment']['clean_control_count']} controls; "
        f"{report['experiment']['synthetic_variant_count']} synthetic variants)."
    )
    for technique, summary in report["experiment"]["techniques"].items():
        print(f"\n{technique}")
        for tokenizer, counts in summary["overall"]["by_tokenizer"].items():
            print(
                f"  {tokenizer}: {counts['original_tokens']} -> "
                f"{counts['optimized_tokens']} tokens; "
                f"{counts['saved_tokens']} saved ({counts['savings_percent']}%)."
            )
    print("No repeated content was removed; semantic evaluation remains unrun.")
    print(f"Dataset saved to {DATASET_FILE}")
    print(f"Results saved to {RESULT_FILE}")
    return report


if __name__ == "__main__":
    main()
