from collections.abc import Iterable
import numpy as np
from copy import deepcopy

from doggos.knowledge.consequents.consequent import Consequent
from doggos.knowledge.clause import Clause
from doggos.fuzzy_sets.fuzzy_set import MembershipDegree


class MamdaniConsequent(Consequent):
    """
    Class used to represent a fuzzy rule consequent in Mamdani model:
    http://researchhubs.com/post/engineering/fuzzy-system/mamdani-fuzzy-model.html

    Attributes
    --------------------------------------------
    __clause : Clause
        supplies Consequent with universe and a fuzzy set

    __cut_clause : Clause
        Clause with fuzzy set cut to the rule firing level

    Methods
    --------------------------------------------
    output(rule_firing: MembershipDegree) -> Clause
        Clause with fuzzy set cut to the rule firing level

    Examples:
    --------------------------------------------
    Creating decision variable
    >>> watering = LinguisticVariable('watering', Domain(0, 1, 0.001))
    Creating fuzzy set for watering
    >>> watering_low = Type1FuzzySet(gaussian(0.5, 0.1))
    Creating clause: watering is low
    >>> watering_low_clause = Clause(watering, 'Low', watering_low)
    Creating MamdaniConsequent
    >>> watering_low_consequent = MamdaniConsequent(watering_low_clause)
    Cutting membership function of low watering to firing level of rule
    >>> firing = 0.3
    >>> cut_clause = watering_low_consequent.output(firing)
    """

    def __init__(self, clause: Clause):
        """
        Create Rules Consequent used in Mamdani Inference System. Provided Clause supplies Consequent with universe
        and a fuzzy set.
        :param clause: Clause containing fuzzy set and linguistic variable
        """
        self.__clause = clause
        self.__cut_clause = None

    @property
    def clause(self) -> Clause:
        """
        Getter of the clause
        :return: clause
        """
        return self.__clause

    @property
    def cut_clause(self) -> Clause:
        """
        Getter of clause with cut membership function. Is None if output method was not called.
        :return: cut clause
        """
        return self.__cut_clause

    def output(self, rule_firing: MembershipDegree) -> Clause:
        """
        Cuts membership function to the level of rule firing. It is a minimum of membership function values
        and respecting rule firing. Rule firing should hold values from range [0, 1]. In case of interval type 2 fuzzy
        sets its recommended to pass a two element collection of floats from range [0, 1].
        IMPORTANT:
        Make sure type of fuzzy set used in Clause matches type of fuzzy sets used in Antecedent of Rule and therefore
        its firing type.
        :param rule_firing: firing value of a Rule in which Consequent is used
        :return: Clause with fuzzy set cut to the rule firing level
        """
        if isinstance(rule_firing, float):
            return self.__cut(rule_firing)
        elif isinstance(rule_firing, Iterable):
            if not isinstance(rule_firing, np.ndarray):
                rule_firing = np.array(rule_firing).reshape((len(rule_firing), 1))
            if rule_firing.shape != (len(rule_firing), 1):
                rule_firing = rule_firing.reshape(len(rule_firing), 1)
            return self.__cut(rule_firing)
        else:
            raise ValueError(f"Incorrect type of rule firing: {rule_firing}")

    def __cut(self, rule_firing: np.ndarray or float) -> Clause:
        """
        Makes a cut for type one fuzzy sets. If Clause fuzzy set type mismatches rule_firing type, exception is raised.
        :param rule_firing: crisp value of rule firing
        :return: Clause with fuzzy set cut to the rule firing level
        """
        self.__cut_clause = deepcopy(self.__clause)
        self.__cut_clause.values = np.minimum(self.__cut_clause.values, rule_firing)
        return self.__cut_clause
