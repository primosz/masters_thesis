from abc import ABC, abstractmethod
from typing import NewType, Iterable

MembershipDegree = NewType('MembershipDegree', None)
"""
MembershipDegree is a type hint for membership degree of some value to a certain fuzzy set. Membership degree type
varies depending on type of fuzzy set used. Therefore we use one type hint for all types for fuzzy sets, so treat it
only as a hint, its not a type.
E.g.:
For type one fuzzy set membership degree is a float.
For type two fuzzy set membership degree is a Tuple[float, float]
For other types of fuzzy sets is a membership degree defined by user.
"""


class FuzzySet(ABC):

    @abstractmethod
    def __call__(self, x: float or Iterable[float]) -> MembershipDegree:
        pass
