import numpy as np

from doggos.algebras.algebra import Algebra, validate_input, expand_negation_argument
from doggos.fuzzy_sets.fuzzy_set import MembershipDegree


class LukasiewiczAlgebra(Algebra):

    @staticmethod
    @validate_input
    def implication(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Lukasiewicz implication
        :param a: first value
        :param b: second value
        :return: min(1., 1 - a + b)
        """
        return np.minimum(1., 1 - a + b)

    @staticmethod
    @expand_negation_argument
    def negation(a: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Lukasiewicz negation
        :param a: value
        :return: 1 - a
        """
        return 1 - a

    @staticmethod
    @validate_input
    def s_norm(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Lukasiewicz S-norm
        :param a: first value
        :param b: second value
        :return: min(1, a + b)
        """
        return np.minimum(1., a + b)

    @staticmethod
    @validate_input
    def t_norm(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Lukasiewicz T-norm
        :param a: first value
        :param b: second value
        :return: max(.0, a + b - 1)
        """
        return np.maximum(.0, a + b - 1.0)
