"""Conservative, meaning-preserving native-language sentence cleanup."""
import re
import unicodedata

import tiktoken


TOKENIZERS = {
    "cl100k_base": "cl100k_base",
    "o200k_base": "o200k_base",
    "p50k_base": "p50k_base",
}
SUPPORTED_LANGUAGES = {
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "ur": "Urdu",
    "en": "English",
}
MAX_OPTIMIZER_INPUT_LENGTH = 10_000
SAFE_EDGE_INVISIBLES = {"\u200b", "\ufeff"}


def _is_word_character(char: str, in_word: bool, next_char: str) -> bool:
    category = unicodedata.category(char)
    if category[0] in ("L", "N") or category[0] == "M" and in_word:
        return True
    return (
        char in ("'", "\u2019")
        and in_word
        and bool(next_char)
        and unicodedata.category(next_char)[0] in ("L", "M")
    )


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    start = None
    for index, char in enumerate(text):
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if _is_word_character(char, start is not None, next_char):
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index, text[start:index]))
            start = None
    if start is not None:
        spans.append((start, len(text), text[start:]))
    return spans


def find_duplicate_word_candidates(text: str) -> list[dict]:
    """Find exact adjacent word repeats; candidates always require user confirmation."""
    words = _word_spans(text)
    candidates = []
    for previous, current in zip(words, words[1:]):
        _, previous_end, previous_word = previous
        start, end, word = current
        separator = text[previous_end:start]
        normalized_previous = unicodedata.normalize("NFC", previous_word).casefold()
        normalized_word = unicodedata.normalize("NFC", word).casefold()
        if separator and separator.isspace() and normalized_previous == normalized_word:
            candidates.append({
                "id": f"duplicate-{start}-{end}",
                "word": word,
                "start": start,
                "end": end,
                "separator_start": previous_end,
                "requires_confirmation": True,
                "reason": (
                    "Exact adjacent repetition detected; text alone cannot establish "
                    "whether it is accidental, emphatic, or grammatical."
                ),
            })
    return candidates


def _remove_safe_edge_invisibles(text: str) -> tuple[str, list[dict], list[dict]]:
    output = []
    removed = []
    counts = {}
    removed_indexes = set()
    for index, char in enumerate(text):
        name = unicodedata.name(char, "UNKNOWN")
        safe_position = (
            index == 0
            or index == len(text) - 1
            or (index > 0 and text[index - 1].isspace())
            or (index + 1 < len(text) and text[index + 1].isspace())
        )
        if char in SAFE_EDGE_INVISIBLES and safe_position:
            counts[name] = counts.get(name, 0) + 1
            removed_indexes.add(index)
            continue
        output.append(char)
    for name, count in counts.items():
        removed.append({
            "type": "invisible_format_character",
            "character_name": name,
            "count": count,
            "description": "Removed at a text edge or next to whitespace.",
        })

    detected = []
    for index, char in enumerate(text):
        name = unicodedata.name(char, "")
        is_invisible = (
            unicodedata.category(char) == "Cf"
            or char == "\u00ad"
            or "ZERO WIDTH" in name
            or "GRAPHEME JOINER" in name
            or "VARIATION SELECTOR" in name
        )
        if not is_invisible:
            continue
        action = "removed" if index in removed_indexes else "preserved"
        detected.append({
            "codepoint": f"U+{ord(char):04X}",
            "name": name or "UNKNOWN",
            "action": action,
        })
    return "".join(output), removed, detected


def _format_text(text: str) -> tuple[str, list[dict]]:
    normalized = unicodedata.normalize("NFC", text)
    changes = []
    if normalized != text:
        changes.append({
            "type": "unicode_nfc",
            "description": "Applied canonical Unicode NFC normalization.",
        })

    whitespace_source = normalized
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    original_lines = normalized.split("\n")
    lines = []
    for line in original_lines:
        line = re.sub(r"[^\S\n]+", " ", line)
        lines.append(line.strip(" "))
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()

    if normalized != unicodedata.normalize("NFC", text):
        replacements = _whitespace_examples(whitespace_source)
        changes.append({
            "type": "formatting_whitespace",
            "description": (
                "Trimmed outer/line-edge whitespace, collapsed repeated horizontal "
                "whitespace, normalized line endings, or compacted excessive blank lines."
            ),
            "examples": replacements,
        })
    return normalized, changes


def _whitespace_examples(text: str) -> list[dict[str, str]]:
    replacements = []

    def add(before: str, after: str) -> None:
        item = {"before": before, "after": after}
        if before != after and item not in replacements and len(replacements) < 10:
            replacements.append(item)

    leading = re.match(r"\s+", text)
    if leading:
        add(leading.group(0), "")
    trailing = re.search(r"\s+$", text)
    if trailing:
        add(trailing.group(0), "")
    for match in re.finditer(r"[^\S\r\n]{2,}", text):
        add(match.group(0), " ")
    for match in re.finditer(r"(?m)^[ \t]+(?=\S)|[ \t]+(?=\r?$)", text):
        add(match.group(0), "")
    for match in re.finditer(r"(?:\r?\n[ \t]*){3,}", text):
        add(match.group(0), "\n\n")
    if "\r\n" in text or "\r" in text:
        add("\r\n" if "\r\n" in text else "\r", "\n")
    return replacements


def _remove_confirmed_duplicates(
    text: str,
    candidates: list[dict],
    confirmed_ids: set[str],
) -> tuple[str, list[dict]]:
    selected = [candidate for candidate in candidates if candidate["id"] in confirmed_ids]
    output = text
    changes = []
    for candidate in reversed(selected):
        separator_start = candidate["separator_start"]
        output = output[:separator_start] + output[candidate["end"]:]
        changes.append({
            "type": "duplicate_word",
            "candidate_id": candidate["id"],
            "word": candidate["word"],
            "removed_text": candidate["word"],
            "start": candidate["start"],
            "end": candidate["end"],
            "removed_separator": text[separator_start:candidate["start"]],
            "description": "Removed only after explicit user confirmation.",
        })
    return output, list(reversed(changes))


def optimize_user_text(
    text: str,
    language: str,
    encoding: str,
    confirmed_duplicate_ids: list[str] | None = None,
) -> dict:
    """Normalize user text conservatively; ambiguous duplicate words are opt-in only."""
    if not isinstance(text, str):
        raise TypeError("The text field must be a string.")
    if not text.strip():
        raise ValueError("Text must not be empty or whitespace-only.")
    if len(text) > MAX_OPTIMIZER_INPUT_LENGTH:
        raise ValueError(
            f"Text must be no more than {MAX_OPTIMIZER_INPUT_LENGTH} characters."
        )
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        raise ValueError("Text contains an invalid unpaired Unicode surrogate.")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. Choose one of: "
            f"{', '.join(SUPPORTED_LANGUAGES)}."
        )
    if encoding not in TOKENIZERS:
        raise ValueError(
            f"Unsupported encoding '{encoding}'. Choose one of: "
            f"{', '.join(TOKENIZERS)}."
        )
    if confirmed_duplicate_ids is None:
        confirmed_duplicate_ids = []
    if (
        not isinstance(confirmed_duplicate_ids, list)
        or any(not isinstance(candidate_id, str) for candidate_id in confirmed_duplicate_ids)
    ):
        raise TypeError("confirmed_duplicate_ids must be a list of candidate ID strings.")

    candidates = find_duplicate_word_candidates(text)
    candidate_ids = {candidate["id"] for candidate in candidates}
    unknown_ids = set(confirmed_duplicate_ids) - candidate_ids
    if unknown_ids:
        raise ValueError("One or more confirmed duplicate IDs do not match this input.")
    confirmed_ids = set(confirmed_duplicate_ids)

    without_duplicates, duplicate_changes = _remove_confirmed_duplicates(
        text, candidates, confirmed_ids
    )
    without_invisibles, invisible_changes, detected_invisibles = (
        _remove_safe_edge_invisibles(without_duplicates)
    )
    optimized, normalization_changes = _format_text(without_invisibles)
    changes = duplicate_changes + invisible_changes + normalization_changes

    encoder = tiktoken.get_encoding(TOKENIZERS[encoding])
    original_tokens = len(encoder.encode(text, disallowed_special=()))
    optimized_tokens = len(encoder.encode(optimized, disallowed_special=()))
    if optimized_tokens > original_tokens:
        return {
            "original_text": text,
            "optimized_text": text,
            "original_sentence": text,
            "optimized_sentence": text,
            "original_tokens": original_tokens,
            "optimized_tokens": original_tokens,
            "tokens_saved": 0,
            "savings_percentage": 0,
            "repeated_words_detected": [candidate["word"] for candidate in candidates],
            "repeated_words_removed": [],
            "whitespace_changes": [],
            "changes": [{
                "type": "rejected_token_increase",
                "proposed_changes": changes,
                "description": (
                    "The proposed normalized text used more tokens with the selected "
                    "encoding; all changes were rejected."
                ),
            }],
            "candidates": candidates,
            "detected_invisible_characters": detected_invisibles,
            "status": "rejected_token_increase",
            "optimization_status": "rejected_token_increase",
            "semantic_equivalence_proven": False,
            "semantic_validation_status": "not_performed_no_reliable_local_validator",
        }

    tokens_saved = original_tokens - optimized_tokens
    if len(confirmed_ids) < len(candidates):
        status = "review_required"
    elif optimized == text:
        status = "no_safe_optimization"
    elif tokens_saved:
        status = "optimized"
    else:
        status = "normalized_no_token_savings"
    return {
        "original_text": text,
        "optimized_text": optimized,
        "original_sentence": text,
        "optimized_sentence": optimized,
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
        "tokens_saved": tokens_saved,
        "savings_percentage": round(tokens_saved / original_tokens * 100, 2)
        if original_tokens
        else 0,
        "repeated_words_detected": [candidate["word"] for candidate in candidates],
        "repeated_words_removed": [
            change for change in changes if change["type"] == "duplicate_word"
        ],
        "whitespace_changes": [
            change
            for change in changes
            if change["type"] in (
                "formatting_whitespace",
                "unicode_nfc",
                "invisible_format_character",
            )
        ],
        "changes": changes,
        "candidates": candidates,
        "confirmed_duplicate_ids": sorted(confirmed_ids),
        "detected_invisible_characters": detected_invisibles,
        "status": status,
        "optimization_status": status,
        "semantic_equivalence_proven": False,
        "semantic_validation_status": "not_performed_no_reliable_local_validator",
    }


def optimize_native_text(
    text: str,
    language: str,
    encoding: str,
    confirmed_duplicate_ids: list[str] | None = None,
) -> dict:
    """Optimize user text and report measured counts; repeats require confirmation."""
    return optimize_user_text(text, language, encoding, confirmed_duplicate_ids)
