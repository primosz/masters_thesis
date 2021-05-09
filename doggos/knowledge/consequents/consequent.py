from abc import ABC, abstractmethod
from typing import NewType

ConsequentOutput = NewType('ConsequentOutput', None)
"""
Consequent output is a type hint. Is describes various types of output Consequent can return. MamdaniConsequent returns
a Clause with fuzzy set of respective type to the one provided in constructor. TakagiSugenoConsequent returns a float,
which is a computed value of combination of linear functions with parameters provided in constructor. In cases of other
inference systems consequent output should be treated as a hint, it's actual type may vary.
"""


class Consequent(ABC):
    """
    Base class for Consequents of Fuzzy Rules.
    https://en.wikipedia.org/wiki/Fuzzy_rule

    Methods
    --------------------------------------------
    output(*args) -> ConsequentOutput
        calculate output of Consequent
    """

    @abstractmethod
    def output(self, *args) -> ConsequentOutput:
        pass
