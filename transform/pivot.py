from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_ID = "facebook/nllb-200-distilled-600M"
LANG_CODES = {"kn": "kan_Knda", "hi": "hin_Deva", "ta": "tam_Taml"}
NLLB_LANG_CODES = {"kn": "kan_Knda", "hi": "hin_Deva", "ta": "tam_Taml"}


class PivotTranslator:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def translate(self, text: str, source_lang: str) -> str:
        if not text:
            return ""
        self.tokenizer.src_lang = NLLB_LANG_CODES[source_lang]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_new_tokens=512,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def translate_flores(translator: PivotTranslator, flores_root: Path,
                     output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    timings = []
    for lang, dataset_name in LANG_CODES.items():
        source = flores_root / f"{dataset_name}.devtest"
        if not source.exists():
            raise FileNotFoundError(f"Missing FLORES file: {source}")
        rows = []
        for sentence_id, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
            original = line.strip()
            started = time.perf_counter()
            translated = translator.translate(original, lang)
            timings.append(time.perf_counter() - started)
            rows.append({"id": sentence_id, "original": original, "transformed": translated})
        output = output_root / f"flores_{lang}_pivot.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "original", "transformed"])
            writer.writeheader()
            writer.writerows(rows)
    if timings:
        print(f"Average seconds per FLORES sentence: {sum(timings) / len(timings):.3f}")


def translate_belebele(translator: PivotTranslator, output_root: Path,
                       limit: int | None = None) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install with: pip install datasets") from exc

    output_root.mkdir(parents=True, exist_ok=True)
    timings = []
    for lang, subset in LANG_CODES.items():
        dataset = load_dataset("facebook/belebele", subset, split="train")
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        rows = []
        for item in dataset:
            text_fields = [
                item["passage"], item["question"],
                item["choices"][0], item["choices"][1],
                item["choices"][2], item["choices"][3],
            ]
            started = time.perf_counter()
            translated = [translator.translate(text, lang) for text in text_fields]
            timings.append(time.perf_counter() - started)
            rows.append({
                "passage": translated[0],
                "question": translated[1],
                "option1": translated[2],
                "option2": translated[3],
                "option3": translated[4],
                "option4": translated[5],
                "answer": item["answer"],
            })
        output = output_root / f"belebele_{lang}_pivot.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{lang}: {len(rows)} pivot items")

    if timings:
        print(f"Average seconds per Belebele item: {sum(timings) / len(timings):.3f}")


def translate_belebele_local(translator: PivotTranslator, raw_root: Path,
                             output_root: Path, limit: int | None = None) -> None:
    """Translate downloaded Belebele JSONL files without Hugging Face datasets."""
    output_root.mkdir(parents=True, exist_ok=True)
    timings = []
    lang_configs = {"kn": "kan_Knda", "hi": "hin_Deva", "ta": "tam_Taml"}

    for lang, filename in lang_configs.items():
        source = raw_root / f"{filename}.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"Missing Belebele file: {source}")
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            selected = 0
            for item_id, line in enumerate(handle):
                if limit is not None and selected >= limit:
                    break
                item = json.loads(line)
                text_fields = [
                    item["flores_passage"], item["question"],
                    item["mc_answer1"], item["mc_answer2"],
                    item["mc_answer3"], item["mc_answer4"],
                ]
                started = time.perf_counter()
                translated = [translator.translate(text, lang) for text in text_fields]
                timings.append(time.perf_counter() - started)
                rows.append({
                    "id": str(item.get("question_number", item_id)),
                    "passage": translated[0],
                    "question": translated[1],
                    "option1": translated[2],
                    "option2": translated[3],
                    "option3": translated[4],
                    "option4": translated[5],
                    "answer": str(item["correct_answer_num"]),
                })
                selected += 1
        output = output_root / f"belebele_{lang}_pivot.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{lang}: {len(rows)} pivot items")

    if timings:
        print(f"Average seconds per Belebele item: {sum(timings) / len(timings):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flores-root", type=Path,
                        default=Path("data/audit/flores200_dataset/devtest"))
    parser.add_argument("--output", type=Path, default=Path("data/transformed"))
    parser.add_argument("--belebele-limit", type=int, default=None,
                        help="Set 100 for the priority batch")
    parser.add_argument("--skip-flores", action="store_true")
    parser.add_argument("--belebele-local", type=Path)
    args = parser.parse_args()

    translator = PivotTranslator()
    if not args.skip_flores:
        translate_flores(translator, args.flores_root, args.output)
    if args.belebele_limit is not None:
        if args.belebele_local:
            translate_belebele_local(
                translator, args.belebele_local, args.output, args.belebele_limit
            )
        else:
            translate_belebele(translator, args.output, args.belebele_limit)
    if torch.cuda.is_available():
        print(f"Peak GPU VRAM: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GiB")


if __name__ == "__main__":
    main()
