"""Performans metriklerinin toplanması (Performance Aspect)."""

from __future__ import annotations

import time
import tracemalloc
from fnmatch import fnmatch
from typing import Any, Iterable

from ..aspect import Aspect, Proceed
from ..joinpoint import JoinPoint
from ..metrics import CallRecord, MetricsRegistry


class PerformanceAspect(Aspect):
    """Her çağrı için bellek kullanımını ölçer ve tüm metrikleri kaydeder.

    Toplanan metrikler (``MetricsRegistry`` içinde metod bazında):

    * çağrı sayısı, başarılı/hatalı çağrı sayısı, başarı oranı
    * toplam / ortalama / min / maks / standart sapma süre (ms)
    * toplam CPU süresi
    * **tepe bellek** (``tracemalloc``): çağrı sırasında Python'un ayırdığı
      en yüksek ek bellek miktarı (KB) ve çağrı sonundaki net bellek farkı

    ``tracemalloc``'un tepe (peak) değeri tek bir global sayaçtır; iç içe
    çağrılarda dıştaki çağrının tepe değeri kaybolmasın diye bir yığın
    (stack) tutulur ve iç çağrının tepe değeri dıştakine aktarılır.

    ``tracemalloc`` her bellek ayırmayı izlediği için kodu yavaşlatır.
    matplotlib gibi çok sayıda küçük nesne üreten metodlar ``memory_exclude``
    ile bellek ölçümünden hariç tutulabilir (süre ölçümü yine yapılır).

    Süre bilgisi ``TimingAspect``'in ``jp.context``'e yazdığı değerden alınır;
    Timing aspect'i bu metoda uygulanmamışsa kendi ölçümünü kullanır.
    """

    default_order = 20  # Timing'in dışında; böylece Timing'in sonucunu okuyabilir.

    def __init__(
        self,
        registry: MetricsRegistry,
        pointcut: str | tuple[str, ...] = "*",
        *,
        track_memory: bool = True,
        memory_exclude: Iterable[str] = (),
        order: int | None = None,
    ) -> None:
        super().__init__(pointcut, order)
        self.registry = registry
        self.track_memory = track_memory
        self.memory_exclude = tuple(memory_exclude)
        self._stack: list[list[float]] = []  # [başlangıç_bellek, çocukların_tepe_değeri]

    def _memory_start(self) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        self._stack.append([float(current), 0.0])

    def _memory_stop(self) -> tuple[float, float]:
        current, peak = tracemalloc.get_traced_memory()
        start, child_peak = self._stack.pop()
        peak = max(float(peak), child_peak)
        if self._stack:
            self._stack[-1][1] = max(self._stack[-1][1], peak)
        else:
            tracemalloc.stop()  # Ölçülen çağrı kalmadı: izlemenin ek yükünü kaldır.
        return (peak - start) / 1024, (current - start) / 1024

    def _tracks_memory(self, jp: JoinPoint) -> bool:
        if not self.track_memory:
            return False
        return not any(fnmatch(jp.signature, p) or fnmatch(jp.method_name, p) for p in self.memory_exclude)

    def around(self, jp: JoinPoint, proceed: Proceed) -> Any:
        fallback_start = time.perf_counter()
        fallback_cpu = time.process_time()
        track = self._tracks_memory(jp)
        if track:
            self._memory_start()
        success = False
        try:
            result = proceed()
            success = True
            return result
        except Exception as exc:
            jp.exception = exc
            raise
        finally:
            peak_kb, delta_kb = self._memory_stop() if track else (0.0, 0.0)
            wall = jp.context.get("wall_time", time.perf_counter() - fallback_start)
            cpu = jp.context.get("cpu_time", time.process_time() - fallback_cpu)
            started = jp.context.get("started_at", self.registry.relative_time(fallback_start))
            self.registry.record_call(
                CallRecord(
                    call_id=jp.call_id,
                    method=jp.signature,
                    depth=jp.depth,
                    started_at=started,
                    wall_time=wall,
                    cpu_time=cpu,
                    success=success,
                    peak_memory_kb=round(max(peak_kb, 0.0), 2),
                    memory_delta_kb=round(delta_kb, 2),
                    error=None if success else type(jp.exception).__name__,
                )
            )
