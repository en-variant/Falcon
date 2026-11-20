from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LANGS = ("kn", "hi", "ta")


def check_flores(output: Path) -> None:
    for lang in LANGS:
        for strategy in ("native", "romanized", "pivot"):
            path = output / f"flores_{lang}_{strategy}.csv"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 1012, f"{path}: expected 1012 rows, got {len(rows)}"
            assert all(
                row["id"] != "" and row["original"] != "" and row["transformed"] != ""
                for row in rows
            ), f"{path}: blank field"
            print(f"{path}: 1012 rows, 0 blank fields")


def check_belebele(output: Path) -> None:
    required = ("id", "passage", "question", "option1", "option2",
                "option3", "option4", "answer")
    for lang in LANGS:
        for strategy in ("native", "romanized", "pivot"):
            path = output / f"belebele_{lang}_{strategy}.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            assert len(rows) == 900, f"{path}: expected 900 rows, got {len(rows)}"
            assert all(
                all(row.get(field) not in (None, "") for field in required)
                for row in rows
            ), f"{path}: blank field"
            print(f"{path}: 900 rows, 0 blank fields")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/transformed"))
    args = parser.parse_args()
    check_flores(args.output)
    check_belebele(args.output)


if __name__ == "__main__":
    main()
