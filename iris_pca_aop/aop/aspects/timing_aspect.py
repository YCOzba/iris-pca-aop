"""Metod çağrılarının zamanlamasının ölçülmesi (Timing Aspect)."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..aspect import Aspect, Proceed
from ..joinpoint import JoinPoint
from ..metrics import MetricsRegistry


class TimingAspect(Aspect):
    """Her çağrının süresini iki farklı saatle ölçer.

    * **Duvar saati (wall time)** — ``time.perf_counter()``: kullanıcının
      beklediği gerçek süre (I/O, bekleme dâhil). Monoton ve yüksek
      çözünürlüklüdür; ``time.time()`` gibi sistem saati değişiminden etkilenmez.
    * **CPU zamanı** — ``time.process_time()``: işlemcinin bu süreç için
      gerçekten çalıştığı süre. Wall ≫ CPU ise metod I/O'da bekliyor demektir.

    Ölçümler ``jp.context`` içine yazılır (``wall_time``, ``cpu_time``,
    ``started_at``); böylece Logging ve Performance aspect'leri aynı ölçümü
    kullanır. ``slow_threshold_ms`` aşılırsa ``WARNING`` loglanır.
    """

    default_order = 30

    def __init__(
        self,
        logger: logging.Logger,
        registry: MetricsRegistry,
        pointcut: str | tuple[str, ...] = "*",
        *,
        slow_threshold_ms: float = 500.0,
        order: int | None = None,
    ) -> None:
        super().__init__(pointcut, order)
        self.logger = logger
        self.registry = registry
        self.slow_threshold_ms = slow_threshold_ms

    def around(self, jp: JoinPoint, proceed: Proceed) -> Any:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        jp.context["started_at"] = self.registry.relative_time(wall_start)
        try:
            return proceed()
        finally:
            wall = time.perf_counter() - wall_start
            cpu = time.process_time() - cpu_start
            jp.context["wall_time"] = wall
            jp.context["cpu_time"] = cpu
            self.logger.debug(
                "%s   [zamanlama] %s: duvar=%.3f ms, cpu=%.3f ms",
                "  " * jp.depth, jp.signature, wall * 1000, cpu * 1000,
                extra={"event": "timing", "call_id": jp.call_id, "method": jp.signature,
                       "duration_ms": round(wall * 1000, 3), "cpu_ms": round(cpu * 1000, 3)},
            )
            if wall * 1000 > self.slow_threshold_ms:
                message = f"{jp.signature} yavaş çalıştı: {wall * 1000:.1f} ms > eşik {self.slow_threshold_ms:.0f} ms"
                self.registry.record_slow_call(message)
                self.logger.warning(message, extra={"event": "slow_call", "method": jp.signature})
