"""Veri doğrulama kontrolleri (Validation Aspect)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Callable, Mapping, Sequence

from ...exceptions import DataValidationError
from ..aspect import Aspect
from ..joinpoint import JoinPoint
from ..metrics import MetricsRegistry, ValidationRecord


@dataclass(frozen=True)
class Rule:
    """Adlandırılmış bir doğrulama kuralı.

    ``check`` fonksiyonu ``JoinPoint`` alır; kural ihlal edilirse
    ``DataValidationError`` fırlatır, aksi hâlde hiçbir şey döndürmez.
    """

    name: str
    check: Callable[[JoinPoint], None]


RuleMap = Mapping[str, Sequence[Rule]]


class ValidationAspect(Aspect):
    """Metodların girdisini (*pre-condition*) ve çıktısını (*post-condition*) doğrular.

    Kurallar metod deseni -> kural listesi biçiminde verilir::

        ValidationAspect(
            pre={"PCA.fit": [finite_matrix, n_components_in_range]},
            post={"IrisDataLoader.load": [no_missing_values]},
        )

    Bu sayede doğrulama kodu iş metodlarının içine dağılmaz (*Design by
    Contract*'ın AOP ile uygulanması). Pointcut, kural sözlüklerindeki
    anahtarlardan otomatik oluşturulur. Her kontrolün sonucu
    ``MetricsRegistry``'ye yazılır.
    """

    default_order = 50  # En içte: hatası Error Handling/Logging tarafından görülür.

    def __init__(
        self,
        logger: logging.Logger,
        registry: MetricsRegistry,
        *,
        pre: RuleMap | None = None,
        post: RuleMap | None = None,
        order: int | None = None,
    ) -> None:
        self.pre: dict[str, list[Rule]] = {k: list(v) for k, v in (pre or {}).items()}
        self.post: dict[str, list[Rule]] = {k: list(v) for k, v in (post or {}).items()}
        super().__init__(tuple(set(self.pre) | set(self.post)) or ("<none>",), order)
        self.logger = logger
        self.registry = registry

    @staticmethod
    def _rules_for(rule_map: dict[str, list[Rule]], jp: JoinPoint) -> list[Rule]:
        rules: list[Rule] = []
        for pattern, items in rule_map.items():
            if fnmatch(jp.method_name, pattern) or fnmatch(jp.signature, pattern):
                rules.extend(items)
        return rules

    def _run(self, rules: list[Rule], jp: JoinPoint, phase: str) -> None:
        for rule in rules:
            try:
                rule.check(jp)
            except DataValidationError as exc:
                self.registry.record_validation(ValidationRecord(jp.signature, rule.name, phase, False, str(exc)))
                self.logger.warning(
                    "%s  [doğrulama:%s] %s -> İHLAL: %s",
                    "  " * jp.depth, phase, rule.name, exc,
                    extra={"event": "validation_failed", "method": jp.signature, "rule": rule.name},
                )
                exc.context.setdefault("kural", rule.name)
                exc.context.setdefault("asama", phase)
                raise
            self.registry.record_validation(ValidationRecord(jp.signature, rule.name, phase, True))
            self.logger.debug(
                "%s  [doğrulama:%s] %s -> OK",
                "  " * jp.depth, phase, rule.name,
                extra={"event": "validation_passed", "method": jp.signature, "rule": rule.name},
            )

    def before(self, jp: JoinPoint) -> None:
        self._run(self._rules_for(self.pre, jp), jp, "pre")

    def after_returning(self, jp: JoinPoint) -> None:
        self._run(self._rules_for(self.post, jp), jp, "post")
