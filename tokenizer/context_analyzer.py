"""Estimate sentence capacity for a token-limited context window."""

import math
import re
from typing import Optional

if __package__:
    from .token_analyzer import TokenAnalyzer
else:
    from token_analyzer import TokenAnalyzer


def estimate_average_tokens_per_sentence(
    text: str, analyzer: Optional[TokenAnalyzer] = None
) -> float:
    """Measure average tokens per sentence in a sample text.

    English punctuation, Kannada danda punctuation, and line breaks are treated
    as sentence boundaries. Empty fragments are ignored.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if analyzer is None:
        analyzer = TokenAnalyzer()

    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?।\r\n]+", text)
        if sentence.strip()
    ]
    if not sentences:
        raise ValueError("text must contain at least one sentence")

    total_tokens = sum(analyzer.count_tokens(sentence) for sentence in sentences)
    return total_tokens / len(sentences)


def estimate_sentences_fit(
    context_window_tokens: int, average_tokens_per_sentence: float
) -> int:
    """Estimate full sentences that fit in a context window."""
    if isinstance(context_window_tokens, bool) or not isinstance(
        context_window_tokens, int
    ):
        raise TypeError("context_window_tokens must be an integer")
    if context_window_tokens < 0:
        raise ValueError("context_window_tokens cannot be negative")
    if isinstance(average_tokens_per_sentence, bool) or not isinstance(
        average_tokens_per_sentence, (int, float)
    ):
        raise TypeError("average_tokens_per_sentence must be a number")
    if not math.isfinite(average_tokens_per_sentence):
        raise ValueError("average_tokens_per_sentence must be finite")
    if average_tokens_per_sentence <= 0:
        raise ValueError("average_tokens_per_sentence must be greater than zero")

    return math.floor(context_window_tokens / average_tokens_per_sentence)


if __name__ == "__main__":
    sample_text = (
        "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ? "
        "ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ. "
        "Hello, how are you?"
    )
    average_tokens = estimate_average_tokens_per_sentence(sample_text)
    context_size = 1_000
    sentence_estimate = estimate_sentences_fit(context_size, average_tokens)
    print(f"Average tokens per sentence: {average_tokens:.2f}")
    print(f"Estimated sentences in {context_size} tokens: {sentence_estimate}")