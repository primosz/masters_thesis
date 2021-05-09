import numpy as np

from doggos.algebras.algebra import Algebra, validate_input, expand_negation_argument
from doggos.fuzzy_sets.fuzzy_set import MembershipDegree


class GodelAlgebra(Algebra):

    @staticmethod
    @validate_input
    def implication(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Gödel implication
        :param a: first value
        :param b: second value
        :return: max(1 - a, b)
        """
        return np.maximum(1 - a, b)

    @staticmethod
    @expand_negation_argument
    def negation(a: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Gödel negation
        :param a: value
        :return: 1 - a
        """
        return 1 - a

    @staticmethod
    @validate_input
    def s_norm(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Gödel S-norm
        :param a: first value
        :param b: second value
        :return: max(a, b)
        """
        return np.maximum(a, b)

    @staticmethod
    @validate_input
    def t_norm(a: MembershipDegree, b: MembershipDegree) -> MembershipDegree:
        """
        Calculate the Gödel T-norm
        :param a: first value
        :param b: second value
        :return: min(a, b)
        """
        return np.minimum(a, b)
