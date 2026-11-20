"""Generate token comparison charts from a LinguaSmart comparison CSV."""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPHS_DIRECTORY = Path(__file__).resolve().parent / "graphs"
DEFAULT_CSV_NAMES = (
    "linguasmart_comparision.csv",
    "linguasmart_comparison.csv",
)


def find_default_csv() -> Path:
    """Find the requested CSV name, allowing the common spelling variant."""
    for csv_name in DEFAULT_CSV_NAMES:
        csv_path = PROJECT_ROOT / csv_name
        if csv_path.is_file():
            return csv_path

    names = " or ".join(DEFAULT_CSV_NAMES)
    raise FileNotFoundError(
        f"Could not find {names} in {PROJECT_ROOT}. "
        "Pass the CSV path as a command-line argument."
    )


def _save_bar_chart(
    data: pd.DataFrame,
    value_column: str,
    title: str,
    y_axis_label: str,
    output_path: Path,
) -> None:
    """Save a labeled bar chart for one value column."""
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(data["representation"], data[value_column], color="#4472C4")
    axis.set_title(title)
    axis.set_xlabel("Text representation")
    axis.set_ylabel(y_axis_label)
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def generate_graphs(
    csv_path: Path, output_directory: Path = GRAPHS_DIRECTORY
) -> List[Path]:
    """Generate token-count and reduction charts, plus cost when available."""
    csv_path = Path(csv_path)
    data = pd.read_csv(csv_path)

    required_columns = {
        "representation",
        "token_count",
        "token_reduction_percent",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"The CSV is missing required columns: {missing}")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    chart_data = data.copy()
    for column in ("token_count", "token_reduction_percent"):
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
        if chart_data[column].notna().sum() == 0:
            raise ValueError(f"The CSV column {column!r} has no numeric values.")

    generated_paths = []
    token_chart = output_directory / "token_counts.png"
    _save_bar_chart(
        chart_data,
        "token_count",
        "Token Counts by Text Representation",
        "Token count",
        token_chart,
    )
    generated_paths.append(token_chart)

    reduction_chart = output_directory / "token_reduction_percentages.png"
    _save_bar_chart(
        chart_data,
        "token_reduction_percent",
        "Token Reduction by Text Representation",
        "Token reduction (%)",
        reduction_chart,
    )
    generated_paths.append(reduction_chart)

    cost_columns = []
    for column in data.columns:
        if "cost" in column.lower():
            numeric_values = pd.to_numeric(data[column], errors="coerce")
            if numeric_values.notna().any():
                chart_data[column] = numeric_values
                cost_columns.append(column)

    if cost_columns:
        cost_chart = output_directory / "cost_comparison.png"
        figure, axis = plt.subplots(figsize=(8, 5))
        chart_data.set_index("representation")[cost_columns].plot(
            kind="bar", ax=axis
        )
        axis.set_title("Cost by Text Representation")
        axis.set_xlabel("Text representation")
        axis.set_ylabel("Cost (units recorded in CSV)")
        axis.tick_params(axis="x", rotation=15)
        axis.legend(title="Cost column")
        figure.tight_layout()
        figure.savefig(cost_chart, dpi=150)
        plt.close(figure)
        generated_paths.append(cost_chart)
    else:
        print(
            "No cost chart generated: the CSV has no numeric cost data. "
            "Add a numeric cost column for each representation (with currency "
            "or units documented), or provide the model's input price per "
            "million tokens and token counts so costs can be calculated."
        )

    return generated_paths


def main(csv_argument: Optional[str] = None) -> None:
    """Run the chart generator using an optional CSV path."""
    csv_path = Path(csv_argument) if csv_argument else find_default_csv()
    generated_paths = generate_graphs(csv_path)
    for graph_path in generated_paths:
        print(f"Generated: {graph_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create LinguaSmart token and cost comparison charts."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Path to linguasmart_comparision.csv (defaults to the project CSV).",
    )
    arguments = parser.parse_args()
    main(arguments.csv_path)
