from __future__ import annotations
from typing import Callable, NoReturn, Iterable

import numpy as np

from doggos.fuzzy_sets.fuzzy_set import FuzzySet


class Type1FuzzySet(FuzzySet):
    """
    Class used to represent a type I fuzzy set:
    https://en.wikipedia.org/wiki/Fuzzy_set
    Attributes
    --------------------------------------------
    __membership_function : Callable[[float], float]
        membership function, determines degree of membership to a fuzzy set
    Methods
    --------------------------------------------
    __call__(x: float) -> float
        calculate degree of membership of element to a fuzzy set
    Examples:
    --------------------------------------------
    Creating simple type I fuzzy setand calculate degree of belonging
    >>> fuzzy_set = Type1FuzzySet(lambda x: 1)
    >>> fuzzy_set(2)
    1
    Creating type I fuzzy set using numpy functions
    >>> import numpy as np
    >>> def sigmoid(x):
    ...    return 1 / (1 + np.exp(-x))
    ...
    >>> fuzzy_set = Type1FuzzySet(sigmoid)
    >>> fuzzy_set(2.5)
    0.9241
    >>> fuzzy_set([0.0, 2.5])
    array([0.5, 0.9241])
    """

    __membership_function: Callable[[float], float]

    def __init__(self, membership_function: Callable[[float], float]):
        """
        Create type I fuzzy set with given membership function.
        Membership function should return values from range [0, 1], but it is not required in library.
        :param membership_function: membership function of a set
        """
        if not callable(membership_function):
            raise ValueError('Membership function must be callable')
        self.__membership_function = np.vectorize(membership_function)

    def __call__(self, x: float or Iterable[float]) -> float or np.ndarray:
        """
        Calculate the degree of membership to a type I fuzzy set for of an element
        :param x: element of domain
        :return: degree of membership of an element
        """
        return self.__membership_function(x)

    @property
    def membership_function(self) -> Callable[[float], float]:
        """
        Getter of the membership function
        :return: membership function
        """
        return self.__membership_function

    @membership_function.setter
    def membership_function(self, new_membership_function: Callable[[float], float]) -> NoReturn:
        """
        Sets new membership function
        :param new_membership_function: new membership function
        :return: NoReturn
        """
        if not callable(new_membership_function):
            raise ValueError('Membership function must be callable')
        self.__membership_function = new_membership_function
