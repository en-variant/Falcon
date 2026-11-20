"""Adapter for Member B's romanization and pivot transformation functions."""

from __future__ import annotations

import importlib
import logging
import os
from functools import lru_cache
from types import ModuleType

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_module() -> ModuleType | None:
    module_name = os.getenv("TOKEN_TAX_TRANSFORM_MODULE")
    if not module_name:
        return None
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            f"Could not import transform module {module_name!r}: {error}"
        ) from error


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
    try:
        transformed = transform(text, language)
    except Exception as error:
        logger.exception("Member B's %s transform failed", function_name)
        raise RuntimeError(f"{function_name}(text, lang) failed: {error}") from error
    if not isinstance(transformed, str):
        raise TypeError(f"{function_name}(text, lang) must return str.")
    return transformed
