
"""Count text tokens with tiktoken through a replaceable tokenizer backend."""

from typing import Optional, Protocol, Sequence


SUPPORTED_ENCODINGS = ("cl100k_base", "o200k_base")


class TokenizerBackend(Protocol):
    """Small interface that allows other tokenizer libraries to be plugged in."""

    def encode(self, text: str) -> Sequence[int]:
        """Return the token IDs for text."""


class TiktokenBackend:
    """Adapter for a supported tiktoken encoding."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        if encoding_name not in SUPPORTED_ENCODINGS:
            choices = ", ".join(SUPPORTED_ENCODINGS)
            raise ValueError(
                f"Unsupported encoding {encoding_name!r}. Choose from: {choices}."
            )

        try:
            import tiktoken
        except ImportError as error:
            raise ImportError(
                "Token counting requires tiktoken. Install it with 'pip install tiktoken'."
            ) from error

        self._encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> Sequence[int]:
        """Return token IDs from the selected tiktoken encoding."""
        return self._encoding.encode(text)


class TokenAnalyzer:
    """Count tokens using one tokenizer instance for every text representation."""

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        tokenizer: Optional[TokenizerBackend] = None,
    ) -> None:
        self.encoding_name = encoding_name
        self.tokenizer = (
            tokenizer if tokenizer is not None else TiktokenBackend(encoding_name)
        )

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return len(self.tokenizer.encode(text))

    def compare_representations(
        self,
        native_kannada: str,
        romanized_kannada: str,
        english: str,
    ) -> dict:
        """Count three equivalent representations with the same tokenizer."""
        return {
            "native_kannada": self.count_tokens(native_kannada),
            "romanized_kannada": self.count_tokens(romanized_kannada),
            "english": self.count_tokens(english),
        }


if __name__ == "__main__":
    # Sample representations of the same greeting.
    sample_counts = TokenAnalyzer("cl100k_base").compare_representations(
        native_kannada="ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        romanized_kannada="Namaskara, neevu hegiddiri?",
        english="Hello, how are you?",
    )
    for representation, token_count in sample_counts.items():
        print(f"{representation}: {token_count} tokens")
