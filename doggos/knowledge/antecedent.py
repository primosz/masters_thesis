from typing import Dict, Callable
from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.fuzzy_set import MembershipDegree
from doggos.knowledge.clause import Clause

from abc import ABC, abstractmethod


class Antecedent(ABC):
    """
    Base class for representing an antecedent:
    https://en.wikipedia.org/wiki/Fuzzy_set
    
    """

    def __init__(self, algebra: Algebra):
        if not isinstance(algebra, Algebra):
            raise TypeError('algebra must be an Algebra type')
        self.__algebra = algebra

    @property
    @abstractmethod
    def fire(self) -> Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]:
        pass

    @property
    def algebra(self) -> Algebra:
        """
        Getter of the algebra
        :return: algebra
        """
        return self.__algebra
