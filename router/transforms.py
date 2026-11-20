"""Adapter for Member B's romanization and pivot transformation functions."""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from types import ModuleType


@lru_cache(maxsize=1)
def _load_module() -> ModuleType | None:
    module_name = os.getenv("TOKEN_TAX_TRANSFORM_MODULE")
    if not module_name:
        return None
    return importlib.import_module(module_name)


def apply_strategy(text: str, strategy: str, language: str) -> str:
    """Apply a configured Member B function; never silently pass through."""
    if strategy == "native":
        return text
    function_name = {"romanized": "romanize", "pivot": "pivot"}.get(strategy)
    if function_name is None:
        raise ValueError(f"Unsupported strategy: {strategy}")

    module = _load_module()
    if module is None:
        raise RuntimeError(
            "No transform module configured; set TOKEN_TAX_TRANSFORM_MODULE "
            "to a module exposing romanize(text, lang) and pivot(text, lang)."
        )
    transform = getattr(module, function_name, None)
    if not callable(transform):
        raise RuntimeError(
            f"Transform module {module.__name__!r} must define "
            f"{function_name}(text, lang)."
        )
    transformed = transform(text, language)
    if not isinstance(transformed, str):
        raise TypeError(f"{function_name}(text, lang) must return str.")
    return transformed
