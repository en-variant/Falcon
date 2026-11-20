"""Train and evaluate the strategy router using held-out passage groups."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from router.features import extract_features


STRATEGIES = ("native", "romanized", "pivot")
RESULT_COLUMNS = {"id", "lang", "strategy", "correct", "tokens"}
EXAMPLE_COLUMNS = {"id", "passage_id", "lang", "text"}


def choose_target(group: pd.DataFrame) -> str:
    """Choose the cheapest correct strategy, or native when none is correct."""
    correct = group[group["correct"] == 1]
    if correct.empty:
        return "native"
    return str(correct.sort_values(["tokens", "strategy"], kind="stable").iloc[0]["strategy"])


def _read_inputs(results_path: Path, examples_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(results_path)
    examples = pd.read_csv(examples_path)
    missing_results = RESULT_COLUMNS - set(results.columns)
    missing_examples = EXAMPLE_COLUMNS - set(examples.columns)
    if missing_results:
        raise ValueError(f"{results_path} is missing columns: {sorted(missing_results)}")
    if missing_examples:
        raise ValueError(f"{examples_path} is missing columns: {sorted(missing_examples)}")
    if results[["id", "lang", "strategy"]].isna().any().any():
        raise ValueError("results.csv cannot contain empty id, lang, or strategy values.")
    if examples[["id", "lang", "passage_id", "text"]].isna().any().any():
        raise ValueError(
            "examples.csv cannot contain empty id, lang, passage_id, or text values."
        )
    if results[["id", "lang", "strategy"]].astype(str).apply(
        lambda column: column.str.strip().eq("").any()
    ).any():
        raise ValueError("results.csv cannot contain blank id, lang, or strategy values.")

    results["correct"] = pd.to_numeric(results["correct"], errors="raise")
    results["tokens"] = pd.to_numeric(results["tokens"], errors="raise")
    if not results["correct"].isin([0, 1]).all():
        raise ValueError("The correct column must contain only 0 or 1.")
    if not results["tokens"].map(math.isfinite).all() or (results["tokens"] < 0).any():
        raise ValueError("The tokens column must contain non-negative numbers.")
    if not set(results["strategy"]).issubset(STRATEGIES):
        raise ValueError(f"strategy must use only {list(STRATEGIES)}.")
    if not set(results["lang"]).issubset({"en", "kn", "hi", "ta"}):
        raise ValueError("lang must use only en, kn, hi, or ta.")
    if results.duplicated(["id", "lang", "strategy"]).any():
        raise ValueError("Each id/lang/strategy combination must be unique.")
    if examples.duplicated(["id", "lang"]).any():
        raise ValueError("Each example id/lang combination must be unique.")
    if examples[["id", "lang", "passage_id", "text"]].astype(str).apply(
        lambda column: column.str.strip().eq("").any()
    ).any():
        raise ValueError("examples.csv cannot contain blank id, lang, passage_id, or text.")

    result_keys = set(zip(results["id"], results["lang"]))
    example_keys = set(zip(examples["id"], examples["lang"]))
    if result_keys != example_keys:
        raise ValueError(
            "results.csv and examples.csv must contain the same id/lang pairs."
        )
    strategy_counts = results.groupby(["id", "lang"])["strategy"].agg(set)
    expected = set(STRATEGIES)
    if not strategy_counts.map(lambda values: values == expected).all():
        raise ValueError("Every question must have exactly native, romanized, and pivot results.")
    return results, examples


def _build_examples(examples: pd.DataFrame) -> pd.DataFrame:
    feature_rows = pd.DataFrame(
        [extract_features(str(text)) for text in examples["text"]],
        index=examples.index,
    )
    return pd.concat([examples.reset_index(drop=True), feature_rows.reset_index(drop=True)], axis=1)


def _evaluation_frame(results: pd.DataFrame, examples: pd.DataFrame) -> pd.DataFrame:
    targets = [
        {"id": question_id, "lang": language, "target": choose_target(group)}
        for (question_id, language), group in results.groupby(
            ["id", "lang"], sort=False
        )
    ]
    grouped = pd.DataFrame(targets)
    return _build_examples(examples).merge(
        grouped, on=["id", "lang"], how="inner", validate="one_to_one"
    )


def _selected_metrics(
    results: pd.DataFrame, choices: pd.DataFrame, name: str
) -> dict[str, float | str]:
    selected = choices.merge(
        results,
        left_on=["id", "lang", "chosen_strategy"],
        right_on=["id", "lang", "strategy"],
        how="left",
        validate="one_to_one",
    )
    if selected["correct"].isna().any():
        raise ValueError(f"{name} selected a strategy without a matching result row.")
    return {
        "strategy": name,
        "accuracy": float(selected["correct"].mean()),
        "mean_tokens": float(selected["tokens"].mean()),
    }


def _baseline_choices(rows: pd.DataFrame, strategy: str) -> pd.DataFrame:
    return rows[["id", "lang"]].assign(chosen_strategy=strategy)


def evaluate(
    results: pd.DataFrame,
    examples: pd.DataFrame,
    model_path: Path,
    predictions_path: Path,
    test_size: float = 0.30,
    random_state: int = 42,
) -> list[dict[str, float | str]]:
    """Evaluate on unseen passages, save router choices, and persist a final model."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    rows = _evaluation_frame(results, examples)
    if rows["passage_id"].nunique() < 2:
        raise ValueError("At least two distinct passage_id groups are required.")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_indices, test_indices = next(
        splitter.split(rows, groups=rows["passage_id"])
    )
    train = rows.iloc[train_indices]
    test = rows.iloc[test_indices]
    train_passages = set(train["passage_id"])
    test_passages = set(test["passage_id"])
    if train_passages & test_passages:
        raise AssertionError("Passage leakage detected between train and test.")
    if train["target"].nunique() < 2:
        raise ValueError(
            "Training split has fewer than two target strategies; add more diverse data "
            "or choose a different random_state."
        )

    feature_columns = list(extract_features("").keys())
    model = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    model.fit(train[feature_columns], train["target"])

    test_choices = test[["id", "lang"]].copy()
    test_choices["chosen_strategy"] = model.predict(test[feature_columns])
    output_choices = test_choices.assign(
        id=test_choices["lang"].astype(str) + ":" + test_choices["id"].astype(str)
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output_choices[["id", "chosen_strategy"]].to_csv(predictions_path, index=False)

    test_results = results.merge(
        test[["id", "lang"]],
        on=["id", "lang"],
        how="inner",
        validate="many_to_one",
    )
    metrics = [
        _selected_metrics(
            test_results, _baseline_choices(test, strategy), f"always-{strategy}"
        )
        for strategy in STRATEGIES
    ]
    metrics.append(_selected_metrics(test_results, test_choices, "router"))
    oracle = test[["id", "lang", "target"]].rename(
        columns={"target": "chosen_strategy"}
    )
    metrics.append(_selected_metrics(test_results, oracle, "oracle"))

    # Keep the held-out results above, then fit the deployment model on all available data.
    final_model = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    final_model.fit(rows[feature_columns], rows["target"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_columns,
            "strategies": list(STRATEGIES),
            "max_depth": 4,
        },
        model_path,
    )

    print(f"Train passages: {len(train_passages)}")
    print(f"Test passages: {len(test_passages)}")
    print(f"Passage overlap: {len(train_passages & test_passages)}")
    print(pd.DataFrame(metrics).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved held-out choices to {predictions_path}")
    print(f"Saved deployment model to {model_path}")
    return metrics


def generate_synthetic(
    results_path: Path, examples_path: Path, question_count: int = 100
) -> None:
    """Create deterministic smoke-test data with an easy-to-explain optimal rule."""
    if question_count < 20:
        raise ValueError("Use at least 20 synthetic questions for grouped evaluation.")
    results: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    for index in range(question_count):
        short = index % 2 == 0
        text = f"hello short {index}" if short else (
            f"ನಮಸ್ಕಾರ " + ("long mixed-language sample " * 12) + str(index)
        )
        question_id = f"synthetic-{index:04d}"
        examples.append(
            {
                "id": question_id,
                "passage_id": f"passage-{index // 3:04d}",
                "lang": "kn" if not short else "en",
                "text": text,
            }
        )
        token_counts = (
            {"native": 5, "romanized": 8, "pivot": 10}
            if short
            else {"native": 32, "romanized": 16, "pivot": 24}
        )
        for strategy in STRATEGIES:
            results.append(
                {
                    "id": question_id,
                    "lang": "kn" if not short else "en",
                    "strategy": strategy,
                    "correct": 1,
                    "tokens": token_counts[strategy],
                }
            )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(results_path, index=False)
    pd.DataFrame(examples).to_csv(examples_path, index=False)
    print(f"Wrote {len(results)} synthetic strategy rows to {results_path}")
    print(f"Wrote {len(examples)} synthetic examples to {examples_path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("data/eval/results.csv"))
    parser.add_argument("--examples", type=Path, default=Path("data/eval/examples.csv"))
    parser.add_argument("--model", type=Path, default=Path("router/model.joblib"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/router/predictions.csv"),
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    if args.synthetic:
        synthetic_results = Path("data/eval/synthetic_results.csv")
        synthetic_examples = Path("data/eval/synthetic_examples.csv")
        generate_synthetic(synthetic_results, synthetic_examples)
        args.results = synthetic_results
        args.examples = synthetic_examples

    results, examples = _read_inputs(args.results, args.examples)
    evaluate(
        results,
        examples,
        args.model,
        args.predictions,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
