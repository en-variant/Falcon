"""Measure tokenizer fertility for aligned English and Indian-language prompts."""
import json
import unicodedata
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parent
QUESTION_FILE = ROOT / "indian_language_questions.json"
ENGLISH_SAMPLES = [
    "What is the capital of India?",
    "What is the chemical formula of water?",
    "Why do plants need sunlight?",
    "How can water be saved at home?",
    "What are three healthy breakfast foods?",
    "How do you make tea?",
    "What should you take with you on a rainy day?",
    "What is solar energy?",
    "What is the difference between weather and climate?",
    "Why should paper be recycled?",
]
TOKENIZER_CONFIGS = [
    {"name": "cl100k (GPT-4)", "encoding": "cl100k_base"},
    {"name": "o200k (GPT-4o)", "encoding": "o200k_base"},
    {"name": "p50k (GPT-3)", "encoding": "p50k_base"},
]


def count_words(text: str) -> int:
    """Count Unicode letter/number runs; this is not language-specific word segmentation."""
    count = 0
    in_word = False
    for char in text:
        category = unicodedata.category(char)
        if category[0] in ("L", "N"):
            if not in_word:
                count += 1
            in_word = True
        elif category[0] == "M" and in_word:
            continue
        else:
            in_word = False
    return count


def load_samples() -> dict[str, list[dict[str, str]]]:
    dataset = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))
    samples = {
        "English": [
            {"id": f"q{index:02}", "text": text}
            for index, text in enumerate(ENGLISH_SAMPLES, start=1)
        ]
    }
    for language, questions in dataset["languages"].items():
        if len(questions) != len(ENGLISH_SAMPLES):
            raise ValueError(f"{language} has {len(questions)} questions; expected 10")
        samples[language] = [
            {"id": f"q{index:02}", "text": text}
            for index, text in enumerate(questions, start=1)
        ]
    return samples


def run_benchmark() -> dict:
    samples = load_samples()
    encoders = {
        config["name"]: tiktoken.get_encoding(config["encoding"])
        for config in TOKENIZER_CONFIGS
    }
    results = {}

    for language, prompts in samples.items():
        words_total = sum(count_words(prompt["text"]) for prompt in prompts)
        chars_total = sum(len(prompt["text"]) for prompt in prompts)
        language_result = {
            "sample_count": len(prompts),
            "word_count": words_total,
            "char_count": chars_total,
            "word_count_method": (
                "Unicode letter/number runs; combining marks stay with the preceding "
                "run. This is not language-specific morphological segmentation."
            ),
            "tokenizers": {},
        }
        for tokenizer_name, encoder in encoders.items():
            measured_samples = []
            total_tokens = 0
            for prompt in prompts:
                token_count = len(encoder.encode(prompt["text"]))
                word_count = count_words(prompt["text"])
                total_tokens += token_count
                measured_samples.append({
                    **prompt,
                    "word_count": word_count,
                    "char_count": len(prompt["text"]),
                    "token_count": token_count,
                    "fertility": round(token_count / word_count, 3) if word_count else 0,
                })
            language_result["tokenizers"][tokenizer_name] = {
                "token_count": total_tokens,
                "fertility": round(total_tokens / words_total, 3) if words_total else 0,
                "mean_tokens_per_sample": round(total_tokens / len(prompts), 3),
                "samples": measured_samples,
            }
        results[language] = language_result

    english_count = results["English"]["sample_count"]
    for language, language_result in results.items():
        for tokenizer_name, tokenizer_result in language_result["tokenizers"].items():
            english_tokens = results["English"]["tokenizers"][tokenizer_name]["token_count"]
            denominator = english_tokens / english_count
            tokenizer_result["multiplier_vs_english"] = (
                round(tokenizer_result["mean_tokens_per_sample"] / denominator, 3)
                if denominator
                else 0
            )
    return results


def print_report(results: dict) -> None:
    print("\nTOKEN FERTILITY — measured counts on 10 aligned prompts per language")
    for tokenizer in (config["name"] for config in TOKENIZER_CONFIGS):
        print(f"\n{tokenizer}")
        print(f"{'Language':<14} {'Samples':>7} {'Words':>7} {'Tokens':>8} {'Tok/word':>9} {'× English':>10}")
        for language, data in results.items():
            measured = data["tokenizers"][tokenizer]
            print(
                f"{language:<14} {data['sample_count']:>7} {data['word_count']:>7} "
                f"{measured['token_count']:>8} {measured['fertility']:>9} "
                f"{measured['multiplier_vs_english']:>10}"
            )
    print("\nCounts are measured with tiktoken encodings; word counts are approximate Unicode runs.")


def main() -> None:
    results = run_benchmark()
    print_report(results)
    output = ROOT / "benchmark_results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
