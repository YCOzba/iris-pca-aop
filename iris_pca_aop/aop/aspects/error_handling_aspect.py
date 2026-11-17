"""Hata yönetimi (Error Handling Aspect)."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from fnmatch import fnmatch
from typing import Any, Callable, Mapping

from ...exceptions import IrisPCAError, PipelineError
from ..aspect import Aspect, Proceed
from ..joinpoint import JoinPoint
from ..metrics import ErrorRecord, MetricsRegistry

Fallback = Callable[[JoinPoint, BaseException], Any]

_HANDLED_FLAG = "__aop_error_handled__"


class ErrorHandlingAspect(Aspect):
    """Hataları tek bir merkezden yakalar, sınıflandırır ve yönetir.

    Strateji (sırasıyla):

    1. **Yeniden deneme (retry)** — ``retry_on`` listesindeki *geçici* hatalar
       (ör. ``OSError``, ``TimeoutError``) ``max_retries`` kez, üstel bekleme
       (exponential backoff) ile yeniden denenir.
    2. **Yedek davranış (fallback)** — metod adı ``fallbacks`` sözlüğündeki
       bir desene uyuyorsa hata loglanır ve yedek fonksiyonun sonucu
       döndürülür. Örn. bir grafik çizilemezse tüm pipeline çökmez
       (*graceful degradation*).
    3. **Alan hataları** (``IrisPCAError`` ve alt sınıfları) olduğu gibi
       yeniden fırlatılır; bunlar zaten anlamlı mesaj taşır.
    4. **Beklenmeyen hatalar** ``PipelineError`` içine sarmalanır; orijinal hata
       ``__cause__`` olarak korunur (``raise ... from exc``).

    Her hata ``MetricsRegistry``'ye kaydedilir ve traceback ile birlikte
    yalnızca **bir kez** (hatanın doğduğu en içteki metodda) loglanır.
    """

    default_order = 40

    def __init__(
        self,
        logger: logging.Logger,
        registry: MetricsRegistry,
        pointcut: str | tuple[str, ...] = "*",
        *,
        max_retries: int = 2,
        retry_on: tuple[type[BaseException], ...] = (OSError, TimeoutError),
        backoff_seconds: float = 0.05,
        fallbacks: Mapping[str, Fallback] | None = None,
        order: int | None = None,
    ) -> None:
        super().__init__(pointcut, order)
        self.logger = logger
        self.registry = registry
        self.max_retries = max_retries
        self.retry_on = retry_on
        self.backoff_seconds = backoff_seconds
        self.fallbacks: dict[str, Fallback] = dict(fallbacks or {})

    # ------------------------------------------------------------------ yardımcılar
    def _fallback_for(self, jp: JoinPoint) -> Fallback | None:
        for pattern, fallback in self.fallbacks.items():
            if fnmatch(jp.method_name, pattern) or fnmatch(jp.signature, pattern):
                return fallback
        return None

    def _record(self, jp: JoinPoint, exc: BaseException, action: str, attempt: int, handled: bool) -> None:
        self.registry.record_error(
            ErrorRecord(
                timestamp=datetime.now().isoformat(timespec="milliseconds"),
                method=jp.signature,
                error_type=type(exc).__name__,
                message=str(exc),
                handled=handled,
                action=action,
                attempt=attempt,
            )
        )

    def _is_retryable(self, exc: BaseException) -> bool:
        return isinstance(exc, self.retry_on) and not isinstance(exc, IrisPCAError)

    # ------------------------------------------------------------------ advice
    def around(self, jp: JoinPoint, proceed: Proceed) -> Any:
        attempt = 0
        while True:
            attempt += 1
            try:
                return proceed()
            except Exception as exc:
                already_handled = getattr(exc, _HANDLED_FLAG, False)

                # 1) Geçici hata -> yeniden dene
                if not already_handled and self._is_retryable(exc) and attempt <= self.max_retries:
                    wait = self.backoff_seconds * (2 ** (attempt - 1))
                    self._record(jp, exc, "retry", attempt, handled=True)
                    self.logger.warning(
                        "%s! %s geçici hata (%s: %s) -> %d. deneme %.2f sn sonra",
                        "  " * jp.depth, jp.signature, type(exc).__name__, exc, attempt + 1, wait,
                        extra={"event": "retry", "method": jp.signature, "attempt": attempt},
                    )
                    time.sleep(wait)
                    continue

                # 2) Yedek davranış tanımlıysa -> hatayı yut, yedek sonucu döndür
                fallback = self._fallback_for(jp)
                if fallback is not None:
                    self._record(jp, exc, "fallback", attempt, handled=True)
                    self.logger.error(
                        "%s! %s başarısız oldu, yedek davranış uygulanıyor: %s: %s",
                        "  " * jp.depth, jp.signature, type(exc).__name__, exc,
                        exc_info=not already_handled,
                        extra={"event": "fallback", "method": jp.signature},
                    )
                    return fallback(jp, exc)

                if already_handled:
                    raise  # İç metodda zaten loglandı/kaydedildi; tekrar loglama.

                # 3) Alan hatası -> kaydet, logla, aynen fırlat
                if isinstance(exc, IrisPCAError):
                    self._record(jp, exc, "raise", attempt, handled=False)
                    self.logger.error(
                        "%s! %s alan hatası: %s: %s",
                        "  " * jp.depth, jp.signature, type(exc).__name__, exc,
                        extra={"event": "domain_error", "method": jp.signature},
                    )
                    setattr(exc, _HANDLED_FLAG, True)
                    raise

                # 4) Beklenmeyen hata -> PipelineError ile sarmala
                self._record(jp, exc, "wrap", attempt, handled=False)
                self.logger.error(
                    "%s! %s beklenmeyen hata: %s: %s",
                    "  " * jp.depth, jp.signature, type(exc).__name__, exc,
                    exc_info=True,
                    extra={"event": "unexpected_error", "method": jp.signature},
                )
                wrapped = PipelineError(
                    f"{jp.signature} çalışırken beklenmeyen hata oluştu",
                    original=exc,
                    context={"hata_tipi": type(exc).__name__, "deneme": attempt},
                )
                setattr(wrapped, _HANDLED_FLAG, True)
                raise wrapped from exc
