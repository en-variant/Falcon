from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


SCRIPT_MAP = {
    "kn": sanscript.KANNADA,
    "hi": sanscript.DEVANAGARI,
    "ta": sanscript.TAMIL,
}

SCHEME_MAP = {
    "ITRANS": sanscript.ITRANS,
    "ISO": sanscript.ISO,
    "HK": sanscript.HK,
}


def romanize_text(text: str, language: str, scheme: str = "ITRANS") -> str:
    """Romanize Indic text using an explicit script-to-Latin scheme."""
    lang = language.lower()
    code = scheme.upper()
    if lang not in SCRIPT_MAP:
        raise ValueError(f"Unsupported language: {language}")
    if code not in SCHEME_MAP:
        raise ValueError(f"Unsupported scheme: {scheme}")
    if not isinstance(text, str):
        return ""
    return transliterate(text, SCRIPT_MAP[lang], SCHEME_MAP[code])


def deromanize_text(text: str, language: str, scheme: str = "ITRANS") -> str:
    lang = language.lower()
    code = scheme.upper()
    if lang not in SCRIPT_MAP or code not in SCHEME_MAP:
        raise ValueError("Unsupported language or scheme")
    return transliterate(text, SCHEME_MAP[code], SCRIPT_MAP[lang])


def character_match_rate(original: list[str], romanized: list[str], language: str,
                         scheme: str = "ITRANS") -> float:
    if not original:
        return 0.0
    exact = sum(
        1 for original_text, romanized_text in zip(original, romanized)
        if deromanize_text(romanized_text, language, scheme) == original_text
    )
    return 100.0 * exact / len(original)


def process_flores(root: Path, output_root: Path, scheme: str = "ITRANS") -> None:
    names = {
        "kn": "kan_Knda",
        "hi": "hin_Deva",
        "ta": "tam_Taml",
    }
    output_root.mkdir(parents=True, exist_ok=True)

    for lang, dataset_name in names.items():
        source = root / f"{dataset_name}.devtest"
        if not source.exists():
            raise FileNotFoundError(
                f"Missing {source}. Download FLORES devtest files to data/audit/flores200_dataset/devtest/"
            )
        original = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()]
        native = pd.DataFrame({
            "id": range(len(original)),
            "original": original,
            "transformed": original,
        })
        romanized = pd.DataFrame({
            "id": range(len(original)),
            "original": original,
            "transformed": [romanize_text(text, lang, scheme) for text in original],
        })
        native.to_csv(output_root / f"flores_{lang}_native.csv", index=False)
        romanized.to_csv(output_root / f"flores_{lang}_romanized.csv", index=False)
        print(f"{lang}: {len(original)} FLORES rows")


def process_belebele(output_root: Path, scheme: str = "ITRANS") -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install with: pip install datasets") from exc

    output_root.mkdir(parents=True, exist_ok=True)
    lang_configs = {"kn": "kan_Knda", "hi": "hin_Deva", "ta": "tam_Taml"}

    for lang, subset in lang_configs.items():
        dataset = load_dataset("facebook/belebele", subset, split="train")
        native_rows = []
        romanized_rows = []
        for item in dataset:
            native_rows.append({
                "id": str(item.get("question_id", item.get("id", len(native_rows)))),
                "passage": item["passage"],
                "question": item["question"],
                "option1": item["choices"][0],
                "option2": item["choices"][1],
                "option3": item["choices"][2],
                "option4": item["choices"][3],
                "answer": item["answer"],
            })
            romanized_rows.append({
                "id": str(item.get("question_id", item.get("id", len(romanized_rows)))),
                "passage": romanize_text(item["passage"], lang, scheme),
                "question": romanize_text(item["question"], lang, scheme),
                "option1": romanize_text(item["choices"][0], lang, scheme),
                "option2": romanize_text(item["choices"][1], lang, scheme),
                "option3": romanize_text(item["choices"][2], lang, scheme),
                "option4": romanize_text(item["choices"][3], lang, scheme),
                "answer": item["answer"],
            })

        for filename, rows in (
            (f"belebele_{lang}_native.jsonl", native_rows),
            (f"belebele_{lang}_romanized.jsonl", romanized_rows),
        ):
            with (output_root / filename).open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{lang}: {len(native_rows)} Belebele rows")


def process_belebele_local(raw_root: Path, output_root: Path,
                          scheme: str = "ITRANS") -> None:
    """Generate native and Romanized Belebele files from downloaded JSONL files."""
    output_root.mkdir(parents=True, exist_ok=True)
    lang_configs = {"kn": "kan_Knda", "hi": "hin_Deva", "ta": "tam_Taml"}

    for lang, filename in lang_configs.items():
        source = raw_root / f"{filename}.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"Missing Belebele file: {source}")

        native_rows = []
        romanized_rows = []
        with source.open("r", encoding="utf-8") as handle:
            for item_id, line in enumerate(handle):
                item = json.loads(line)
                passage = item["flores_passage"]
                question = item["question"]
                options = [
                    item["mc_answer1"],
                    item["mc_answer2"],
                    item["mc_answer3"],
                    item["mc_answer4"],
                ]
                answer = str(item["correct_answer_num"])

                native_rows.append({
                    "id": str(item.get("question_number", item_id)),
                    "passage": passage,
                    "question": question,
                    "option1": options[0],
                    "option2": options[1],
                    "option3": options[2],
                    "option4": options[3],
                    "answer": answer,
                })
                romanized_rows.append({
                    "id": str(item.get("question_number", item_id)),
                    "passage": romanize_text(passage, lang, scheme),
                    "question": romanize_text(question, lang, scheme),
                    "option1": romanize_text(options[0], lang, scheme),
                    "option2": romanize_text(options[1], lang, scheme),
                    "option3": romanize_text(options[2], lang, scheme),
                    "option4": romanize_text(options[3], lang, scheme),
                    "answer": answer,
                })

        for filename_out, rows in (
            (f"belebele_{lang}_native.jsonl", native_rows),
            (f"belebele_{lang}_romanized.jsonl", romanized_rows),
        ):
            with (output_root / filename_out).open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{lang}: {len(native_rows)} Belebele rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=SCHEME_MAP, default="ITRANS")
    parser.add_argument("--flores-root", type=Path,
                        default=Path("data/audit/flores200_dataset/devtest"))
    parser.add_argument("--output", type=Path, default=Path("data/transformed"))
    parser.add_argument("--belebele", action="store_true")
    parser.add_argument("--belebele-local", type=Path)
    parser.add_argument("--skip-flores", action="store_true")
    args = parser.parse_args()

    if not args.skip_flores:
        process_flores(args.flores_root, args.output, args.scheme)
    if args.belebele_local:
        process_belebele_local(args.belebele_local, args.output, args.scheme)
    elif args.belebele:
        process_belebele(args.output, args.scheme)


if __name__ == "__main__":
    main()
