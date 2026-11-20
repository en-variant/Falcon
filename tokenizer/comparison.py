
"""Compare token counts and export results as a pandas CSV report."""

from pathlib import Path
from typing import Optional

import pandas as pd

if __package__:
    from .cost_calculator import calculate_token_savings
    from .token_analyzer import TokenAnalyzer
else:
    from cost_calculator import calculate_token_savings
    from token_analyzer import TokenAnalyzer


def compare_representations(
    native_kannada: str,
    romanized_kannada: str,
    english: str,
    reference_representation: str = "native_kannada",
    analyzer: Optional[TokenAnalyzer] = None,
) -> pd.DataFrame:
    """Create a comparison table with reductions relative to a chosen baseline.

    Native Kannada is the default reference. If that reference has zero tokens,
    percentage reductions are undefined and are left blank in the report.
    """
    if analyzer is None:
        analyzer = TokenAnalyzer()

    token_counts = analyzer.compare_representations(
        native_kannada, romanized_kannada, english
    )
    if reference_representation not in token_counts:
        names = ", ".join(token_counts)
        raise ValueError(
            f"Unknown reference representation. Choose from: {names}."
        )

    reference_count = token_counts[reference_representation]
    rows = []
    for representation, token_count in token_counts.items():
        if reference_count == 0:
            tokens_saved = reference_count - token_count
            percentage_reduction = None
        else:
            tokens_saved, percentage_reduction = calculate_token_savings(
                reference_count, token_count
            )

        rows.append(
            {
                "representation": representation,
                "token_count": token_count,
                "tokens_saved_vs_reference": tokens_saved,
                "token_reduction_percent": percentage_reduction,
                "reference_representation": reference_representation,
            }
        )

    return pd.DataFrame(rows)


def write_comparison_csv(
    output_path: str,
    native_kannada: str,
    romanized_kannada: str,
    english: str,
    reference_representation: str = "native_kannada",
    analyzer: Optional[TokenAnalyzer] = None,
) -> pd.DataFrame:
    """Write a comparison report to CSV and return its DataFrame."""
    report = compare_representations(
        native_kannada=native_kannada,
        romanized_kannada=romanized_kannada,
        english=english,
        reference_representation=reference_representation,
        analyzer=analyzer,
    )
    report.to_csv(Path(output_path), index=False)
    return report


if __name__ == "__main__":
    # Sample inputs compare the same greeting in three forms.
    sample_report = write_comparison_csv(
        output_path="linguasmart_comparison.csv",
        native_kannada="ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        romanized_kannada="Namaskara, neevu hegiddiri?",
        english="Hello, how are you?",
    )
    print(sample_report.to_string(index=False))
