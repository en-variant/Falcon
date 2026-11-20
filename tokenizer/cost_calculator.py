"""Estimate input costs using prices supplied by the caller."""

import math
from numbers import Real
from typing import Tuple


def _validate_token_count(token_count: int, name: str) -> None:
    if isinstance(token_count, bool) or not isinstance(token_count, int):
        raise TypeError(f"{name} must be an integer")
    if token_count < 0:
        raise ValueError(f"{name} cannot be negative")


def _validate_price(price_per_million_tokens: Real) -> None:
    if isinstance(price_per_million_tokens, bool) or not isinstance(
        price_per_million_tokens, Real
    ):
        raise TypeError("price_per_million_tokens must be a number")
    if not math.isfinite(price_per_million_tokens) or price_per_million_tokens < 0:
        raise ValueError("price_per_million_tokens must be finite and non-negative")


def calculate_input_cost(
    token_count: int, price_per_million_tokens: Real
) -> float:
    """Estimate input cost from an explicitly supplied price per million tokens.

    The returned value uses the same currency as the supplied price. No provider
    price is assumed by this module.
    """
    _validate_token_count(token_count, "token_count")
    _validate_price(price_per_million_tokens)
    return token_count * float(price_per_million_tokens) / 1_000_000


def calculate_token_savings(
    reference_token_count: int, compared_token_count: int
) -> Tuple[int, float]:
    """Return tokens saved and percent reduction versus a reference count.

    A negative saving means the compared text uses more tokens. A zero-token
    reference has no meaningful percentage denominator and raises ValueError.
    """
    _validate_token_count(reference_token_count, "reference_token_count")
    _validate_token_count(compared_token_count, "compared_token_count")
    if reference_token_count == 0:
        raise ValueError("reference_token_count must be greater than zero")

    tokens_saved = reference_token_count - compared_token_count
    percentage_reduction = tokens_saved / reference_token_count * 100
    return tokens_saved, percentage_reduction


def calculate_cost_savings(
    reference_token_count: int,
    compared_token_count: int,
    price_per_million_tokens: Real,
) -> float:
    """Return estimated cost saved versus a reference count."""
    reference_cost = calculate_input_cost(
        reference_token_count, price_per_million_tokens
    )
    compared_cost = calculate_input_cost(
        compared_token_count, price_per_million_tokens
    )
    return reference_cost - compared_cost


if __name__ == "__main__":
    # The sample token counts are fixed; the price comes from your configuration.
    configured_price = float(
        input("Enter your configured input price per million tokens: ")
    )
    reference_tokens = 1_000
    compared_tokens = 800
    saved_tokens, reduction_percent = calculate_token_savings(
        reference_tokens, compared_tokens
    )
    estimated_cost = calculate_input_cost(compared_tokens, configured_price)
    print(f"Estimated input cost: {estimated_cost}")
    print(f"Tokens saved: {saved_tokens} ({reduction_percent:.1f}% reduction)")