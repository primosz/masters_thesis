from __future__ import annotations

from typing import Dict, Sequence, Callable, NoReturn
from doggos.algebras.algebra import Algebra
from doggos.fuzzy_sets.fuzzy_set import MembershipDegree
from doggos.knowledge.clause import Clause
from doggos.knowledge.antecedent import Antecedent


class Term(Antecedent):
    """
    Class representing an antecedent with recursive firing value computation:
    https://en.wikipedia.org/wiki/Fuzzy_set
    Attributes
    --------------------------------------------
    __clause : Clause
        clause which is stored in antecedent
    __algebra : Algebra
        algebra provides t-norm and s-norm
    Methods
    --------------------------------------------
    def fire(self) -> Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]
        returns a firing value of the antecedent
    Examples
    --------------------------------------------
    TODO
    """

    def __init__(self, algebra: Algebra, clause: Clause = None):
        """
        Creates Term object with given algebra and clause.
        :param algebra: algebra provides t-norm and s-norm
        :param clause: provides a linguistic variable with corresponding fuzzy set
        """
        super().__init__(algebra)
        if not clause:
            self.__fire = None
        else:
            self.__fire = lambda dict_: dict_[clause]

    @property
    def fire(self) -> Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]:
        """
        Returns the firing function
        :return: firing function
        """
        return self.__fire

    @fire.setter
    def fire(self, fire: Callable[[Dict[Clause, MembershipDegree]], MembershipDegree]):
        """
        Sets new firing function to the antecedent
        :param fire: firing function
        """
        self.__fire = fire

    def __and__(self, term: Term) -> Term:
        """
        Creates new antecedent object and sets new firing function which uses t-norm.
        :param term: other term
        :return: term
        """
        new_term = self.__class__(self.algebra)
        new_term.fire = lambda dict_: self.algebra.t_norm(self.fire(dict_), term.fire(dict_))
        return new_term

    def __or__(self, term: Term) -> Term:
        """
        Creates new antecedent object and sets new firing function which uses s-norm.
        :param term: other term
        :return: term
        """
        new_term = self.__class__(self.algebra)
        new_term.fire = lambda dict_: self.algebra.s_norm(self.fire(dict_), term.fire(dict_))
        return new_term
