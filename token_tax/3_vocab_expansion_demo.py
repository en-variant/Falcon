"""Train and hold out-test a standalone multilingual SentencePiece BPE tokenizer."""
import json
import hashlib
import tempfile
from pathlib import Path

import sentencepiece as spm
import tiktoken


ROOT = Path(__file__).resolve().parent
QUESTION_FILE = ROOT / "indian_language_questions.json"
OUTPUT_FILE = ROOT / "vocab_expansion_results.json"
TRAINING_QUESTION_COUNT = 8
VOCAB_SIZE = 1024
_TRAINED_PROCESSOR = None
_TRAINED_CORPUS_SHA256 = None


def _train_tokenizer(texts: list[str], model_prefix: Path):
    corpus_path = model_prefix.with_suffix(".txt")
    corpus_path.write_text("\n".join(texts), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        hard_vocab_limit=False,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        minloglevel=2,
    )
    processor = spm.SentencePieceProcessor()
    if not processor.load(str(model_prefix) + ".model"):
        raise RuntimeError("SentencePiece failed to load the trained model.")
    return processor


def _load_training_corpus() -> tuple[list[str], str]:
    dataset_bytes = QUESTION_FILE.read_bytes()
    dataset = json.loads(dataset_bytes.decode("utf-8"))
    train_texts = []
    for language, questions in dataset["languages"].items():
        if len(questions) <= TRAINING_QUESTION_COUNT:
            raise ValueError(f"{language} does not have held-out questions.")
        train_texts.extend(questions[:TRAINING_QUESTION_COUNT])
    return train_texts, hashlib.sha256(dataset_bytes).hexdigest()


def get_trained_processor():
    """Return a tokenizer trained on the current training split for held-out/custom inputs."""
    global _TRAINED_PROCESSOR, _TRAINED_CORPUS_SHA256
    train_texts, corpus_sha256 = _load_training_corpus()
    if _TRAINED_PROCESSOR is not None and _TRAINED_CORPUS_SHA256 == corpus_sha256:
        return _TRAINED_PROCESSOR
    with tempfile.TemporaryDirectory(prefix="token-tax-spm-") as temp_dir:
        processor = _train_tokenizer(train_texts, Path(temp_dir) / "indic_bpe")
    _TRAINED_PROCESSOR = processor
    _TRAINED_CORPUS_SHA256 = corpus_sha256
    return processor


def evaluate_custom_text(text: str) -> dict:
    if not isinstance(text, str):
        raise TypeError("Custom text must be a string.")
    if len(text) > 100_000:
        raise ValueError("Custom text must be no more than 100,000 characters.")
    processor = get_trained_processor()
    piece_ids = processor.encode(text, out_type=int)
    pieces = processor.encode(text, out_type=str)
    round_trip = processor.decode(piece_ids)
    base_count = len(tiktoken.get_encoding("cl100k_base").encode(
        text, disallowed_special=()
    ))
    return {
        "text": text,
        "sentencepiece_piece_count": len(piece_ids),
        "sentencepiece_pieces": pieces,
        "cl100k_token_count": base_count,
        "piece_count_change_percent_vs_cl100k": round(
            (len(piece_ids) / base_count - 1) * 100, 2
        ) if base_count else 0,
        "round_trip": round_trip,
        "round_trip_exact": round_trip == text,
        "tokenizer": "separately trained SentencePiece BPE; not a GPT or Gemini vocabulary",
        "semantic_or_answer_quality_evaluated": False,
    }


def run_vocab_expansion() -> dict:
    dataset_bytes = QUESTION_FILE.read_bytes()
    dataset = json.loads(dataset_bytes.decode("utf-8"))
    test_by_language = {}
    for language, questions in dataset["languages"].items():
        if len(questions) <= TRAINING_QUESTION_COUNT:
            raise ValueError(f"{language} does not have held-out questions.")
        test_by_language[language] = questions[TRAINING_QUESTION_COUNT:]

    encoder = tiktoken.get_encoding("cl100k_base")
    train_texts, corpus_sha256 = _load_training_corpus()
    processor = get_trained_processor()
    results = {}
    overall_base_tokens = 0
    overall_piece_count = 0
    total_holdout_samples = 0

    for language, texts in test_by_language.items():
        measured = []
        base_total = 0
        piece_total = 0
        exact_round_trips = 0
        for index, text in enumerate(texts, start=TRAINING_QUESTION_COUNT + 1):
            base_count = len(encoder.encode(text, disallowed_special=()))
            piece_ids = processor.encode(text, out_type=int)
            pieces = processor.encode(text, out_type=str)
            round_trip = processor.decode(piece_ids)
            exact = round_trip == text
            base_total += base_count
            piece_total += len(piece_ids)
            exact_round_trips += exact
            measured.append({
                "id": f"q{index:02}",
                "text": text,
                "cl100k_tokens": base_count,
                "sentencepiece_pieces": len(piece_ids),
                "pieces": pieces,
                "round_trip": round_trip,
                "round_trip_exact": exact,
            })
        overall_base_tokens += base_total
        overall_piece_count += piece_total
        total_holdout_samples += len(texts)
        results[language] = {
            "training_sample_count": TRAINING_QUESTION_COUNT,
            "heldout_sample_count": len(texts),
            "heldout_cl100k_tokens": base_total,
            "heldout_sentencepiece_pieces": piece_total,
            "piece_count_change_percent": round(
                (piece_total / base_total - 1) * 100, 2
            )
            if base_total
            else 0,
            "exact_round_trips": exact_round_trips,
            "samples": measured,
        }

    return {
        "experiment": {
            "dataset_version": dataset["dataset_version"],
            "model_type": "SentencePiece BPE",
            "requested_vocab_size": VOCAB_SIZE,
            "actual_vocab_size": processor.get_piece_size(),
            "training_sample_count": len(train_texts),
            "training_corpus_sha256": corpus_sha256,
            "heldout_sample_count": total_holdout_samples,
            "split": (
                "Questions q01-q08 per language train the tokenizer; remaining "
                "questions are held out and are not used in training."
            ),
            "normalization": "identity",
            "baseline": "cl100k_base",
            "baseline_tokens": overall_base_tokens,
            "sentencepiece_pieces": overall_piece_count,
            "piece_count_change_percent": round(
                (overall_piece_count / overall_base_tokens - 1) * 100, 2
            )
            if overall_base_tokens
            else 0,
            "exact_round_trips": sum(
                value["exact_round_trips"] for value in results.values()
            ),
            "production_claim": (
                "This is a separately trained tokenizer, not a vocabulary merge "
                "into cl100k or a deployed language model. Piece-count differences "
                "do not establish answer quality or production token savings."
            ),
        },
        "languages": results,
    }


def main() -> None:
    report = run_vocab_expansion()
    print("\nSENTENCEPIECE — multilingual BPE, held-out evaluation")
    print(
        f"Train: {report['experiment']['training_sample_count']} samples; "
        f"test: {report['experiment']['heldout_sample_count']} samples; "
        f"actual vocabulary: {report['experiment']['actual_vocab_size']} pieces."
    )
    for language, result in report["languages"].items():
        print(
            f"{language}: {result['heldout_cl100k_tokens']} cl100k tokens, "
            f"{result['heldout_sentencepiece_pieces']} SentencePiece pieces; "
            f"change {result['piece_count_change_percent']}%; "
            f"exact round trips {result['exact_round_trips']}/{result['heldout_sample_count']}."
        )
    OUTPUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
