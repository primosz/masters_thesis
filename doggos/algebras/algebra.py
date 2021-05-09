from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

from doggos.fuzzy_sets.fuzzy_set import MembershipDegree


def validate_input(function):
    """
    Decorator for functions in algebras that checks dimensions. If the input is iterable, it is converted to np.ndarray.
    For the operation to make sense, the first dimension must match. If one input is a float and the other is an array,
    for example, the operation continues if the dimensions match.

    :param function: operation, that takes two arguments a and b, for example: implication, t_norm, s_norm
    :return: decorated negation
    """

    def operation(a, b):
        if isinstance(a, Iterable):
            a = np.array(a)
            size_a = a.shape[0]
        else:
            size_a = 1
        if isinstance(b, Iterable):
            b = np.array(b)
            size_b = b.shape[0]
        else:
            size_b = 1
        if size_a != size_b:
            raise ValueError(f'Dimensions {size_a} and {size_b} are not compatible')
        return function(a, b)

    return operation


def expand_negation_argument(negation):
    """
    Expand argument dimensions for negation.
    For example passing: [0.1, 0.2] allows to calculate negation and returns an array of size 2.
    :param negation: negation function, takes one argument
    :return: decorated function
    """

    def operation(a):
        if isinstance(a, Iterable):
            a = np.array(a)
        return negation(a)

    return operation


class Algebra(ABC):
    """
    Class that represents algebra for specific fuzzy logic.
    For example: Lukasiewicz algebra, Gödel algebra.
    Each algebra contains following operations:
    - T-norm: generalized AND
    - S-norm: generalized OR
    - Negation
    - Implication
    """

    @staticmethod
    @abstractmethod
    def t_norm(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        pass

    @staticmethod
    @abstractmethod
    def s_norm(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        pass

    @staticmethod
    @abstractmethod
    def negation(a: MembershipDegree) -> MembershipDegree:
        pass

    @staticmethod
    @abstractmethod
    def implication(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        pass
