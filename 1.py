"""Compatibility entry point for Member B Romanization."""

from transform.romanize import romanize_text


if __name__ == "__main__":
    test_data = {
        "kn": "ನಮಸ್ಕಾರ",
        "hi": "नमस्ते",
        "ta": "வணக்கம்",
    }
    for language, text in test_data.items():
        print(language, romanize_text(text, language, scheme="ITRANS"))
