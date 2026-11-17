"""Aspect temel sınıfı.

Bir aspect, programın birçok yerine dağılmış olan *kesişen ilgiyi*
(cross-cutting concern: loglama, zamanlama, doğrulama, hata yönetimi...)
tek bir sınıfta toplar. Aspect'in **ne zaman** çalışacağını *pointcut*,
**ne yapacağını** ise *advice* metodları belirler.

Desteklenen advice türleri (AspectJ ile aynı isimlendirme):

* ``before``          : metod çalışmadan önce
* ``after_returning`` : metod başarıyla döndükten sonra
* ``after_throwing``  : metod hata fırlattığında
* ``after``           : her durumda (``finally``)
* ``around``          : metodu tamamen sarmalar; çağırıp çağırmamaya,
                        tekrar denemeye veya sonucu değiştirmeye karar verebilir
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any, Callable, Iterable

from .joinpoint import JoinPoint

Proceed = Callable[[], Any]


class Aspect:
    """Tüm aspect'lerin türediği temel sınıf.

    Parameters
    ----------
    pointcut:
        Aspect'in uygulanacağı metod adları için glob desenleri
        (ör. ``"*"``, ``"fit*"``, ``("load_data", "preprocess")``).
    order:
        Zincirdeki sıra. **Küçük değer = dışta** çalışır. Örneğin loglama
        aspect'i en dışta olmalıdır ki diğer aspect'lerin ürettiği hataları da
        görebilsin.
    """

    #: Sınıf düzeyinde varsayılan sıra; alt sınıflar değiştirebilir.
    default_order: int = 100

    def __init__(self, pointcut: str | Iterable[str] = "*", order: int | None = None) -> None:
        self.pointcut: tuple[str, ...] = (pointcut,) if isinstance(pointcut, str) else tuple(pointcut)
        self.order: int = self.default_order if order is None else order

    # ------------------------------------------------------------------ pointcut
    def matches(self, class_name: str, method_name: str) -> bool:
        """Metod bu aspect'in pointcut'ına uyuyor mu?

        Desenler hem yalın metod adıyla (``fit*``) hem de ``Sinif.metod``
        biçimindeki tam adla (``PCA.fit*``) karşılaştırılır.
        """
        signature = f"{class_name}.{method_name}"
        return any(fnmatch(method_name, p) or fnmatch(signature, p) for p in self.pointcut)

    # ------------------------------------------------------------------ advices
    def before(self, jp: JoinPoint) -> None:  # pragma: no cover - varsayılan boş
        """Metod çalışmadan önce çağrılır."""

    def after_returning(self, jp: JoinPoint) -> None:  # pragma: no cover
        """Metod başarıyla tamamlandığında çağrılır (``jp.result`` doludur)."""

    def after_throwing(self, jp: JoinPoint) -> None:  # pragma: no cover
        """Metod hata fırlattığında çağrılır (``jp.exception`` doludur)."""

    def after(self, jp: JoinPoint) -> None:  # pragma: no cover
        """Başarı veya hata fark etmeksizin en son çağrılır."""

    def around(self, jp: JoinPoint, proceed: Proceed) -> Any:
        """Varsayılan *around* advice'ı diğer dört advice'ı sırasıyla uygular.

        Alt sınıflar daha fazla kontrol gerektiğinde (ör. yeniden deneme)
        bu metodu ezebilir. ``proceed()`` zincirdeki bir sonraki aspect'i,
        en sonda da asıl metodu çalıştırır.
        """
        self.before(jp)
        try:
            result = proceed()
        except Exception as exc:
            jp.exception = exc
            self.after_throwing(jp)
            raise
        else:
            jp.result = result
            jp.exception = None
            self.after_returning(jp)
            return result
        finally:
            self.after(jp)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(pointcut={self.pointcut!r}, order={self.order})"
