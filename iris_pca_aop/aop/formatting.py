"""Loglarda argüman ve sonuçları kısa/okunur biçimde göstermek için yardımcılar.

Büyük bir NumPy dizisini veya DataFrame'i loga olduğu gibi yazmak hem logu
okunmaz hâle getirir hem de diski şişirir. Bu yüzden yalnızca tip ve boyut
bilgisi yazılır.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

_MAX_REPR = 60


def summarize(value: Any) -> str:
    """Bir değerin tek satırlık özetini döndürür."""
    if isinstance(value, pd.DataFrame):
        return f"DataFrame{value.shape}"
    if isinstance(value, pd.Series):
        return f"Series(len={len(value)})"
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}:{value.dtype}"
    if isinstance(value, dict):
        return f"dict(keys={len(value)})"
    if isinstance(value, (list, tuple)) and len(value) > 4:
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, tuple):
        return "(" + ", ".join(summarize(v) for v in value) + ")"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return f"{type(value).__name__}(...)"
    if type(value).__repr__ is object.__repr__:
        return f"<{type(value).__name__}>"
    text = repr(value)
    return text if len(text) <= _MAX_REPR else text[: _MAX_REPR - 3] + "..."


def summarize_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Bir çağrının argümanlarını ``a, b, k=v`` biçiminde özetler."""
    parts = [summarize(a) for a in args]
    parts += [f"{k}={summarize(v)}" for k, v in kwargs.items()]
    return ", ".join(parts)
