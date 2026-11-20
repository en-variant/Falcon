"""Train and cross-validate the strategy router on passage-grouped data."""

from __future__ import annotations

import argparse
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from router.features import SUPPORTED_LANGUAGES, extract_routing_features


STRATEGIES = ("native", "romanized", "pivot")
MODEL_VERSION = 1
RESULT_COLUMNS = {"id", "lang", "strategy", "correct", "tokens"}
EXAMPLE_COLUMNS = {"id", "passage_id", "lang", "text"}


def choose_target(group: pd.DataFrame) -> str:
    """Choose the cheapest correct strategy, or native when none is correct."""
    correct = group[group["correct"] == 1]
    if correct.empty:
        return "native"
    return str(
        correct.sort_values(["tokens", "strategy"], kind="stable").iloc[0]["strategy"]
    )


def _read_inputs(
    results_path: Path, examples_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(results_path, dtype={"id": "string", "lang": "string"})
    examples = pd.read_csv(
        examples_path,
        dtype={"id": "string", "lang": "string", "passage_id": "string"},
    )
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
    if examples[["id", "lang", "passage_id", "text"]].astype(str).apply(
        lambda column: column.str.strip().eq("").any()
    ).any():
        raise ValueError(
            "examples.csv cannot contain blank id, lang, passage_id, or text."
        )

    results["correct"] = pd.to_numeric(results["correct"], errors="raise")
    results["tokens"] = pd.to_numeric(results["tokens"], errors="raise")
    if not results["correct"].isin([0, 1]).all():
        raise ValueError("The correct column must contain only 0 or 1.")
    if not results["tokens"].map(math.isfinite).all() or (results["tokens"] < 0).any():
        raise ValueError("The tokens column must contain finite, non-negative numbers.")
    if not set(results["strategy"]).issubset(STRATEGIES):
        raise ValueError(f"strategy must use only {list(STRATEGIES)}.")
    if not set(results["lang"]).issubset(SUPPORTED_LANGUAGES):
        raise ValueError(f"lang must use only {list(SUPPORTED_LANGUAGES)}.")
    if results.duplicated(["id", "lang", "strategy"]).any():
        raise ValueError("Each id/lang/strategy combination must be unique.")
    if examples.duplicated(["id", "lang"]).any():
        raise ValueError("Each example id/lang combination must be unique.")
    if examples["id"].duplicated().any():
        raise ValueError(
            "Prediction contract requires globally unique IDs because predictions.csv "
            "does not include a lang column."
        )

    result_keys = set(zip(results["id"], results["lang"]))
    example_keys = set(zip(examples["id"], examples["lang"]))
    if result_keys != example_keys:
        raise ValueError("results.csv and examples.csv must contain the same id/lang pairs.")
    strategy_counts = results.groupby(["id", "lang"])["strategy"].agg(set)
    expected = set(STRATEGIES)
    if not strategy_counts.map(lambda values: values == expected).all():
        raise ValueError(
            "Every question must have exactly native, romanized, and pivot results."
        )
    return results, examples


def _build_examples(examples: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(
        [
            extract_routing_features(str(text), str(language))
            for text, language in zip(examples["text"], examples["lang"])
        ],
        index=examples.index,
    )
    return pd.concat(
        [examples.reset_index(drop=True), features.reset_index(drop=True)], axis=1
    )


def _evaluation_frame(results: pd.DataFrame, examples: pd.DataFrame) -> pd.DataFrame:
    targets = [
        {"id": question_id, "lang": language, "target": choose_target(group)}
        for (question_id, language), group in results.groupby(
            ["id", "lang"], sort=False
        )
    ]
    return _build_examples(examples).merge(
        pd.DataFrame(targets),
        on=["id", "lang"],
        how="inner",
        validate="one_to_one",
    )


def _bootstrap_intervals(
    selected: pd.DataFrame, random_state: int, n_samples: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    group_stats = (
        selected.groupby("passage_id", sort=False)
        .agg(
            correct_sum=("correct", "sum"),
            token_sum=("tokens", "sum"),
            question_count=("id", "size"),
        )
        .reset_index(drop=True)
    )
    if len(group_stats) < 2:
        raise ValueError("At least two held-out passage groups are required for CIs.")

    rng = np.random.default_rng(random_state)
    counts = rng.multinomial(
        len(group_stats),
        np.full(len(group_stats), 1 / len(group_stats)),
        size=n_samples,
    )
    sizes = group_stats["question_count"].to_numpy()
    denominators = counts @ sizes
    accuracies = (counts @ group_stats["correct_sum"].to_numpy()) / denominators
    mean_tokens = (counts @ group_stats["token_sum"].to_numpy()) / denominators
    accuracy_ci = np.quantile(accuracies, [0.025, 0.975])
    token_ci = np.quantile(mean_tokens, [0.025, 0.975])
    return (
        (float(accuracy_ci[0]), float(accuracy_ci[1])),
        (float(token_ci[0]), float(token_ci[1])),
    )


def _selected_metrics(
    results: pd.DataFrame,
    rows: pd.DataFrame,
    choices: pd.DataFrame,
    name: str,
    random_state: int,
    bootstrap_samples: int,
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
    selected = selected.merge(
        rows[["id", "lang", "passage_id"]],
        on=["id", "lang"],
        how="left",
        validate="one_to_one",
    )
    accuracy_ci, token_ci = _bootstrap_intervals(
        selected, random_state, bootstrap_samples
    )
    return {
        "strategy": name,
        "accuracy": float(selected["correct"].mean()),
        "accuracy_ci_low": accuracy_ci[0],
        "accuracy_ci_high": accuracy_ci[1],
        "mean_tokens": float(selected["tokens"].mean()),
        "tokens_ci_low": token_ci[0],
        "tokens_ci_high": token_ci[1],
    }


def _baseline_choices(rows: pd.DataFrame, strategy: str) -> pd.DataFrame:
    return rows[["id", "lang"]].assign(chosen_strategy=strategy)


def _data_fingerprint(results: pd.DataFrame, examples: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for frame, sort_columns in (
        (results, ["id", "lang", "strategy"]),
        (examples, ["id", "lang"]),
    ):
        canonical = frame.sort_values(sort_columns).to_csv(index=False).encode("utf-8")
        digest.update(canonical)
    return digest.hexdigest()


def evaluate(
    results: pd.DataFrame,
    examples: pd.DataFrame,
    model_path: Path,
    predictions_path: Path,
    random_state: int = 42,
    n_splits: int = 5,
    bootstrap_samples: int = 1000,
    training_data_kind: str = "unverified",
) -> list[dict[str, float | str]]:
    """Run grouped out-of-fold evaluation and persist a full-data serving model."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100.")
    if training_data_kind not in {"real", "synthetic", "unverified"}:
        raise ValueError("training_data_kind must be real, synthetic, or unverified.")

    rows = _evaluation_frame(results, examples)
    passage_count = rows["passage_id"].nunique()
    if passage_count < n_splits:
        raise ValueError(
            f"Need at least {n_splits} distinct passage_id groups for grouped "
            f"cross-validation; found {passage_count}."
        )
    feature_columns = list(extract_routing_features("", "en").keys())
    splitter = GroupKFold(n_splits=n_splits)
    out_of_fold: list[pd.DataFrame] = []
    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(rows, groups=rows["passage_id"]), start=1
    ):
        train = rows.iloc[train_indices]
        test = rows.iloc[test_indices]
        overlap = set(train["passage_id"]) & set(test["passage_id"])
        if overlap:
            raise AssertionError(f"Passage leakage in fold {fold_number}: {overlap}")
        if train["target"].nunique() < 2:
            raise ValueError(
                f"Training fold {fold_number} contains fewer than two target "
                "strategies. Collect more varied real examples or reduce --folds."
            )
        fold_model = DecisionTreeClassifier(max_depth=4, random_state=random_state)
        fold_model.fit(train[feature_columns], train["target"])
        fold_choices = test[["id", "lang"]].copy()
        fold_choices["chosen_strategy"] = fold_model.predict(test[feature_columns])
        out_of_fold.append(fold_choices)

    router_choices = pd.concat(out_of_fold, ignore_index=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    router_choices[["id", "chosen_strategy"]].to_csv(predictions_path, index=False)
    metrics = [
        _selected_metrics(
            results,
            rows,
            _baseline_choices(rows, strategy),
            f"always-{strategy}",
            random_state,
            bootstrap_samples,
        )
        for strategy in STRATEGIES
    ]
    metrics.append(
        _selected_metrics(
            results,
            rows,
            router_choices,
            "router",
            random_state,
            bootstrap_samples,
        )
    )
    oracle = rows[["id", "lang", "target"]].rename(
        columns={"target": "chosen_strategy"}
    )
    metrics.append(
        _selected_metrics(
            results,
            rows,
            oracle,
            "oracle",
            random_state,
            bootstrap_samples,
        )
    )

    final_model = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    final_model.fit(rows[feature_columns], rows["target"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_columns,
            "strategies": list(STRATEGIES),
            "max_depth": 4,
            "model_version": MODEL_VERSION,
            "training_data_kind": training_data_kind,
            "training_rows": len(rows),
            "training_passages": passage_count,
            "data_sha256": _data_fingerprint(results, examples),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        model_path,
    )
    print(f"Grouped CV folds: {n_splits}")
    print(f"Data rows: {len(rows)}")
    print(f"Passage groups: {passage_count}")
    print("Maximum train/test passage overlap: 0")
    print(pd.DataFrame(metrics).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved out-of-fold choices to {predictions_path}")
    print(f"Saved deployment model to {model_path}")
    return metrics


def generate_synthetic(
    results_path: Path, examples_path: Path, question_count: int = 100
) -> None:
    """Create deterministic pipeline-test rows; never use these as experiment data."""
    if question_count < 20:
        raise ValueError("Use at least 20 synthetic questions for grouped evaluation.")
    results: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    for index in range(question_count):
        short = index % 2 == 0
        text = f"hello short {index}" if short else (
            "ನಮಸ್ಕಾರ " + ("long mixed-language sample " * 12) + str(index)
        )
        question_id = f"synthetic-{index:04d}"
        language = "en" if short else "kn"
        examples.append(
            {
                "id": question_id,
                "passage_id": f"passage-{index // 3:04d}",
                "lang": language,
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
                    "lang": language,
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
    parser.add_argument("--results", type=Path)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--confirm-real-data",
        action="store_true",
        help="Confirm results/examples are real experiment data, not pipeline fixtures.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    if args.synthetic and args.confirm_real_data:
        parser.error("--synthetic and --confirm-real-data cannot be combined.")
    if not args.synthetic and not args.confirm_real_data:
        parser.error(
            "Real training requires --confirm-real-data after checking both CSV inputs."
        )

    if args.synthetic:
        args.results = args.results or Path("data/eval/synthetic_results.csv")
        args.examples = args.examples or Path("data/eval/synthetic_examples.csv")
        args.model = args.model or Path("router/synthetic_model.joblib")
        args.predictions = args.predictions or Path("data/router/synthetic_predictions.csv")
        generate_synthetic(args.results, args.examples)
        data_kind = "synthetic"
    else:
        args.results = args.results or Path("data/eval/results.csv")
        args.examples = args.examples or Path("data/eval/examples.csv")
        args.model = args.model or Path("router/model.joblib")
        args.predictions = args.predictions or Path("data/router/predictions.csv")
        data_kind = "real"

    results, examples = _read_inputs(args.results, args.examples)
    evaluate(
        results,
        examples,
        args.model,
        args.predictions,
        random_state=args.random_state,
        n_splits=args.folds,
        bootstrap_samples=args.bootstrap_samples,
        training_data_kind=data_kind,
    )


if __name__ == "__main__":
    main()
