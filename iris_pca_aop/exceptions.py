"""Projeye özgü istisna (exception) hiyerarşisi.

Tüm alan (domain) hataları ``IrisPCAError`` sınıfından türetilir. Böylece
hata yönetimi aspect'i, beklenen (alan) hataları ile beklenmeyen (sistem)
hatalarını birbirinden ayırt edebilir.
"""

from __future__ import annotations

from typing import Any


class IrisPCAError(Exception):
    """Projedeki tüm özel hataların temel sınıfı."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context or {})

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} ({details})"


class DataLoadError(IrisPCAError):
    """Veri seti okunamadığında / yüklenemediğinde fırlatılır."""


class DataValidationError(IrisPCAError):
    """Bir metodun girdisi veya çıktısı doğrulama kurallarını geçemediğinde fırlatılır."""


class PCAError(IrisPCAError):
    """PCA modeli ile ilgili hatalar (ör. model eğitilmeden dönüşüm yapılması)."""


class VisualizationError(IrisPCAError):
    """Grafik üretimi sırasında oluşan hatalar."""


class PipelineError(IrisPCAError):
    """Beklenmeyen bir sistem hatasının alan hatasına sarmalanmış hâli."""

    def __init__(
        self,
        message: str,
        *,
        original: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.original = original
