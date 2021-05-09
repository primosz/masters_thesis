from abc import ABC, abstractmethod

import collections.abc as abc
from typing import Iterable

from doggos.knowledge import Rule


class InferenceSystem(ABC):
    _rule_base: Iterable[Rule]

    def __init__(self, rule_base: Iterable[Rule]):
        if not isinstance(rule_base, abc.Iterable) or any(not isinstance(rule, Rule) for rule in rule_base):
            raise TypeError('rule_base must be an iterable of type Rule')

        self._rule_base = rule_base

    @abstractmethod
    def infer(self, *args) -> Iterable[float]:
        pass


