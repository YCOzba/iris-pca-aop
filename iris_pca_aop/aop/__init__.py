"""Hafif, bağımlılıksız bir Aspect Oriented Programming (AOP) çatısı.

Temel kavramlar:

* :class:`JoinPoint` — araya girilebilen nokta (metod çağrısı)
* :class:`Aspect`    — kesişen ilgiyi (logging, timing...) kapsülleyen sınıf
* pointcut           — aspect'in hangi metodlara uygulanacağını seçen desen
* advice             — before / after_returning / after_throwing / after / around
* :func:`weave`      — aspect'leri bir sınıfa çalışma anında uygular
"""

from .aspect import Aspect
from .joinpoint import JoinPoint
from .metrics import MetricsRegistry
from .weaver import advise, is_woven, weave

__all__ = ["Aspect", "JoinPoint", "MetricsRegistry", "advise", "is_woven", "weave"]
