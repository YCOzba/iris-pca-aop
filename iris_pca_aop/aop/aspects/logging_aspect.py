"""İşlem adımlarının loglanması (Logging Aspect)."""

from __future__ import annotations

import logging

from ..aspect import Aspect
from ..formatting import summarize, summarize_call
from ..joinpoint import JoinPoint


class LoggingAspect(Aspect):
    """Her metod çağrısının başlangıcını, bitişini ve hatasını loglar.

    * Başlangıç  : ``INFO``  — adım açıklaması (docstring'in ilk satırı) + argümanlar
    * Bitiş      : ``INFO``  — süre (``TimingAspect`` ölçtüyse) + sonucun özeti
    * Hata       : ``ERROR`` — hata tipi ve mesajı

    İç içe çağrılar girintiyle gösterilir; böylece log dosyası, pipeline'ın
    çağrı ağacını okunur biçimde yansıtır. Her log kaydına ``extra`` ile
    yapılandırılmış alanlar (``event``, ``call_id``, ``method``...) eklenir;
    JSON log dosyası bu alanları ayrı sütunlar olarak saklar.
    """

    default_order = 10  # En dışta: diğer tüm aspect'lerin etkisini görür.

    def __init__(
        self,
        logger: logging.Logger,
        pointcut: str | tuple[str, ...] = "*",
        *,
        log_arguments: bool = True,
        log_results: bool = True,
        order: int | None = None,
    ) -> None:
        super().__init__(pointcut, order)
        self.logger = logger
        self.log_arguments = log_arguments
        self.log_results = log_results

    @staticmethod
    def _indent(jp: JoinPoint) -> str:
        return "  " * jp.depth

    def _extra(self, jp: JoinPoint, event: str, **fields: object) -> dict[str, object]:
        return {
            "event": event,
            "call_id": jp.call_id,
            "method": jp.signature,
            "depth": jp.depth,
            "step": jp.description,
            **fields,
        }

    def before(self, jp: JoinPoint) -> None:
        args = f" | girdi: {summarize_call(jp.args, jp.kwargs)}" if self.log_arguments and (jp.args or jp.kwargs) else ""
        self.logger.info(
            "%s-> [#%d] %s başladı: %s%s",
            self._indent(jp), jp.call_id, jp.signature, jp.description, args,
            extra=self._extra(jp, "start"),
        )

    def after_returning(self, jp: JoinPoint) -> None:
        wall = jp.context.get("wall_time")
        duration = f" ({wall * 1000:.2f} ms)" if wall is not None else ""
        result = f" | çıktı: {summarize(jp.result)}" if self.log_results and jp.result is not None else ""
        self.logger.info(
            "%s<- [#%d] %s tamamlandı%s%s",
            self._indent(jp), jp.call_id, jp.signature, duration, result,
            extra=self._extra(jp, "end", duration_ms=None if wall is None else round(wall * 1000, 3)),
        )

    def after_throwing(self, jp: JoinPoint) -> None:
        exc = jp.exception
        self.logger.error(
            "%sx- [#%d] %s BAŞARISIZ: %s: %s",
            self._indent(jp), jp.call_id, jp.signature, type(exc).__name__, exc,
            extra=self._extra(jp, "error", error_type=type(exc).__name__),
        )
