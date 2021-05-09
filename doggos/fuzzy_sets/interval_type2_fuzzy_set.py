from __future__ import annotations
from typing import Callable, Iterable, NoReturn

import numpy as np

from doggos.fuzzy_sets.fuzzy_set import FuzzySet


class IntervalType2FuzzySet(FuzzySet):
    """
    Class used to represent a interval type II fuzzy set:

    https://en.wikipedia.org/wiki/Fuzzy_set

    Attributes
    --------------------------------------------
    __upper_membership_function: Callable[[float], float]
        upper membership function, determines the upper membership degree to a fuzzy set
    __lower_membership_function: Callable[[float], float]
        lower membership function, determines the lower membership degree to a fuzzy set

    Methods
    --------------------------------------------
    __call__(x: float) -> Tuple[float, float]
        calculate the membership degree to a fuzzy set of an element

    Examples:
    --------------------------------------------
    Creating interval type II fuzzy set using numpy functions (suggested)
    >>> import numpy as np
    >>> def f1(x):
    ...    return 1 / (1 + np.exp(-x))
    ...
    >>>def f2(x):
    ...    return 1
    ...
    >>> fuzzy_set = IntervalType2FuzzySet(f1, f2)
    >>> fuzzy_set(2.5)
    (0.9241, 1)
    >>> fuzzy_set([0.0, 2.5])
    (array([0.5, 1]), array([0.9241, 1]))
    """
    __upper_membership_function: Callable[[float], float]
    __lower_membership_function: Callable[[float], float]

    def __init__(self,
                 lower_membership_function: Callable[[float], float],
                 upper_membership_function: Callable[[float], float]):
        """
        Create interval type II fuzzy set with given lower membership function and upper membership function.
        Both functions should return values from range [0, 1].
        IMPORTANT:
        Lower membership function should return lower respective values.
        This is not validated in constructor, but it will raise an exception if you try to call a fuzzy set with
        incorrect functions.
        :param upper_membership_function: upper membership function of a set
        :param lower_membership_function: lower membership function of a set
        """
        if not callable(upper_membership_function) or not callable(lower_membership_function):
            raise ValueError('Membership functions should be callable')
        self.__upper_membership_function = np.vectorize(upper_membership_function)
        self.__lower_membership_function = np.vectorize(lower_membership_function)

    def __call__(self, x: float or Iterable[float]) -> np.ndarray:
        """
        Calculate the degree of belonging (lower_membership, upper_membership),
        raises an exception if lower_membership > upper_membership

        :param x: element of domain
        :return: membership degree of an element as tuple (lmf(x), umf(x))
        """
        lower_membership, upper_membership = self.__lower_membership_function(x), self.__upper_membership_function(x)
        if np.any(lower_membership > upper_membership):
            raise ValueError('Lower membership function returned higher value than upper membership function')
        return np.array([lower_membership, upper_membership])

    @property
    def upper_membership_function(self) -> Callable[[float], float]:
        """
        Getter of the upper membership function
        :return: upper membership function
        """
        return self.__upper_membership_function

    @upper_membership_function.setter
    def upper_membership_function(self, new_upper_membership_function: Callable[[float], float]) -> NoReturn:
        """
        Setter of the upper membership function
        :param new_upper_membership_function: new upper membership function, must be callable
        :return: NoReturn
        """
        if not callable(new_upper_membership_function):
            raise ValueError('Membership function should be callable')
        self.__upper_membership_function = new_upper_membership_function

    @property
    def lower_membership_function(self) -> Callable[[float], float]:
        """
        Getter of the lower membership function
        :return: lower membership function
        """
        return self.__lower_membership_function

    @lower_membership_function.setter
    def lower_membership_function(self, new_lower_membership_function: Callable[[float], float]) -> NoReturn:
        """
        Setter of the lower membership function
        :param new_lower_membership_function: new lower membership function, must be callable
        :return: NoReturn
        """
        if not callable(new_lower_membership_function):
            raise ValueError('Membership function should be callable')
        self.__lower_membership_function = new_lower_membership_function
