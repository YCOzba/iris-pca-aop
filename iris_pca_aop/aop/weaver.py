"""Weaver (dokuyucu): aspect'leri hedef sınıflara/fonksiyonlara uygular.

Weaving, aspect kodunun asıl iş koduna *eklenmesi* işlemidir. Java'da
AspectJ bunu derleme anında yapar; Python'da ise çalışma anında (runtime)
metodları sarmalayan (wrapper) fonksiyonlar üreterek yapıyoruz.

Önemli tasarım kararı: ``weave`` orijinal sınıfı **değiştirmez**; onun
yerine aspect'lerle donatılmış bir **alt sınıf** üretir. Böylece iş
sınıfları (``IrisDataLoader``, ``PCA`` ...) loglama/zamanlama kodundan
tamamen habersiz kalır ve test edilirken aspect'siz hâlleri de kullanılabilir.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
from typing import Any, Callable, Iterable, Sequence, TypeVar

from .aspect import Aspect
from .joinpoint import JoinPoint

T = TypeVar("T", bound=type)

#: İç içe çağrı derinliği (loglarda girinti için). ``contextvars`` kullanıldığı
#: için thread/async güvenlidir.
_call_depth: contextvars.ContextVar[int] = contextvars.ContextVar("aop_call_depth", default=0)

#: Bir metodun zaten dokunduğunu işaretlemek için kullanılan öznitelik.
_WOVEN_FLAG = "__aop_woven__"


def _build_chain(
    func: Callable[..., Any],
    aspects: Sequence[Aspect],
    class_name: str,
    method_name: str,
    bound: bool,
) -> Callable[..., Any]:
    """Verilen fonksiyonu, aspect zinciri ile sarmalayan wrapper'ı üretir."""
    doc = inspect.getdoc(func) or ""
    description = doc.strip().splitlines()[0].rstrip(".") if doc.strip() else method_name

    @functools.wraps(func)
    def advised(*args: Any, **kwargs: Any) -> Any:
        target = args[0] if bound and args else None
        call_args = tuple(args[1:]) if bound else tuple(args)
        depth = _call_depth.get()
        jp = JoinPoint(
            target=target,
            class_name=class_name,
            method_name=method_name,
            args=call_args,
            kwargs=dict(kwargs),
            depth=depth,
            description=description,
        )

        def invoke(index: int) -> Any:
            if index == len(aspects):
                # Zincirin sonu: asıl metodu çalıştır.
                if bound:
                    return func(target, *jp.args, **jp.kwargs)
                return func(*jp.args, **jp.kwargs)
            return aspects[index].around(jp, lambda: invoke(index + 1))

        token = _call_depth.set(depth + 1)
        try:
            return invoke(0)
        finally:
            _call_depth.reset(token)

    setattr(advised, _WOVEN_FLAG, True)
    advised.__aop_aspects__ = tuple(aspects)  # type: ignore[attr-defined]
    return advised


def _sorted_for(class_name: str, method_name: str, aspects: Iterable[Aspect]) -> list[Aspect]:
    """Metoda uyan aspect'leri ``order`` değerine göre (dıştan içe) sıralar."""
    return sorted((a for a in aspects if a.matches(class_name, method_name)), key=lambda a: a.order)


def weave(cls: T, aspects: Iterable[Aspect], *, include_private: bool = False) -> T:
    """``cls`` sınıfının public metodlarına aspect'leri uygular.

    Parameters
    ----------
    cls:
        Hedef sınıf. Değiştirilmez; aspect'li bir alt sınıf döndürülür.
    aspects:
        Uygulanacak aspect nesneleri. Her aspect yalnızca kendi
        pointcut'ına uyan metodlara uygulanır.
    include_private:
        ``True`` ise ``_`` ile başlayan metodlar da dokunur
        (``__dunder__`` metodlar hiçbir zaman dokunmaz).
    """
    aspects = list(aspects)
    advised_members: dict[str, Any] = {}

    for klass in reversed(cls.__mro__[:-1]):  # object hariç, kalıtım zinciri dâhil
        for name, member in vars(klass).items():
            if name.startswith("__") or (name.startswith("_") and not include_private):
                continue
            if isinstance(member, (staticmethod, classmethod, property)):
                continue
            if not inspect.isfunction(member):
                continue
            chain = _sorted_for(cls.__name__, name, aspects)
            if chain:
                advised_members[name] = _build_chain(member, chain, cls.__name__, name, bound=True)

    woven = type(cls.__name__, (cls,), advised_members)
    woven.__qualname__ = cls.__qualname__
    woven.__module__ = cls.__module__
    woven.__doc__ = cls.__doc__
    woven.__aop_target__ = cls  # type: ignore[attr-defined]
    return woven  # type: ignore[return-value]


def advise(*aspects: Aspect) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Tek bir fonksiyona aspect uygulamak için dekoratör.

    Örnek::

        @advise(TimingAspect(), LoggingAspect())
        def hesapla(x): ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        module = func.__module__.rsplit(".", 1)[-1]
        chain = _sorted_for(module, func.__name__, aspects)
        return _build_chain(func, chain, module, func.__name__, bound=False)

    return decorator


def is_woven(func: Any) -> bool:
    """Bir fonksiyonun/metodun aspect'lerle sarmalanıp sarmalanmadığını döndürür."""
    return bool(getattr(func, _WOVEN_FLAG, False))
