"""Fast, deterministic text features for the strategy router."""

from __future__ import annotations

import re
import string


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_INDIC_RANGES = {
    "kn": ((0x0C80, 0x0CFF),),
    "hi": ((0x0900, 0x097F),),
    "ta": ((0x0B80, 0x0BFF),),
}
_INDIC_ALL_RANGES = tuple(
    codepoint_range
    for ranges in _INDIC_RANGES.values()
    for codepoint_range in ranges
)
_PUNCTUATION = set(string.punctuation) | set("।॥、，。！？；：…—–“”‘’")


def _in_ranges(character: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in ranges)


def infer_language(text: str) -> str:
    """Infer the dominant supported Indic script, defaulting to English."""
    counts = {
        language: sum(_in_ranges(char, ranges) for char in text)
        for language, ranges in _INDIC_RANGES.items()
    }
    language, count = max(counts.items(), key=lambda item: item[1])
    return language if count else "en"


def extract_features(text: str) -> dict[str, float]:
    """Return the contract's zero-model features as numeric values."""
    characters = list(text)
    denominator = max(len(characters), 1)
    word_count = len(_WORD_RE.findall(text))
    indic_count = sum(_in_ranges(char, _INDIC_ALL_RANGES) for char in characters)
    latin_count = sum(char.isascii() and char.isalpha() for char in characters)
    digit_count = sum(char.isdigit() for char in characters)
    punctuation_count = sum(char in _PUNCTUATION for char in characters)
    url_matches = _URL_RE.findall(text)

    return {
        "char_length": float(len(characters)),
        "word_count": float(word_count),
        "indic_proportion": indic_count / denominator,
        "latin_proportion": latin_count / denominator,
        "digit_proportion": digit_count / denominator,
        "punctuation_proportion": punctuation_count / denominator,
        "url_proportion": len(url_matches) / max(word_count, 1),
    }
