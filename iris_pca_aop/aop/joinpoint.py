"""Join point (birleşim noktası) modeli.

AOP terminolojisinde *join point*, programın akışında bir aspect'in araya
girebileceği noktadır. Bu projede join point'ler **metod çağrılarıdır**.
Her çağrı için bir ``JoinPoint`` nesnesi oluşturulur ve zincirdeki tüm
aspect'lere aynı nesne iletilir; böylece aspect'ler çağrı hakkındaki bilgiyi
(argümanlar, sonuç, hata, süre...) birbirleriyle paylaşabilir.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

_call_counter = itertools.count(1)


@dataclass
class JoinPoint:
    """Tek bir metod çağrısını temsil eder."""

    target: Any
    """Metodun çağrıldığı nesne (``self``)."""

    class_name: str
    method_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    depth: int = 0
    """İç içe (nested) çağrı derinliği; 0 = en dış çağrı."""

    description: str = ""
    """Metodun docstring'inin ilk satırı; loglarda *işlem adımı* olarak yazılır."""

    call_id: int = field(default_factory=lambda: next(_call_counter))
    result: Any = None
    exception: BaseException | None = None
    context: dict[str, Any] = field(default_factory=dict)
    """Aspect'lerin birbirine bilgi aktarabileceği serbest alan."""

    @property
    def signature(self) -> str:
        """``Sinif.metod`` biçiminde, loglarda kullanılan kısa ad."""
        return f"{self.class_name}.{self.method_name}"

    def get_argument(self, name: str, position: int, default: Any = None) -> Any:
        """Bir argümanı adına (kwargs) ya da sırasına (args) göre döndürür."""
        if name in self.kwargs:
            return self.kwargs[name]
        if len(self.args) > position:
            return self.args[position]
        return default
