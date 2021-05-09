import numpy as np
import collections.abc as abc

from typing import List, Dict, Tuple, Sequence, Callable, NoReturn, Iterable


from doggos.fuzzy_sets import MembershipDegree
from doggos.knowledge.clause import Clause
from doggos.knowledge.rule import Rule
from doggos.knowledge.consequents.mamdani_consequent import MamdaniConsequent
from doggos.inference.inference_system import InferenceSystem
from doggos.fuzzy_sets.type1_fuzzy_set import Type1FuzzySet


class MamdaniInferenceSystem(InferenceSystem):
    """
    Class used to represent a mamdani inference system:

    https://www.mathworks.com/help/fuzzy/types-of-fuzzy-inference-systems.html

    Attributes
    --------------------------------------------
    _rule_base: Iterable[Rule]
        fuzzy rule base used for inference

    Methods
    --------------------------------------------
    infer(self, defuzzification_method: Callable, features: Dict[Clause, List[MembershipDegree]])
            -> Iterable[float] or float:
        infer decision from rule base

    Examples:
    --------------------------------------------
    Creating simple mamdani inference system and infering decision
    >>> rule_base = [first_rule, second_rule, third_rule]
    >>> features: Dict[Clause, MembershipDegree] = fuzzifier.fuzzify(dataset)
    >>> mamdani = MamdaniInferenceSystem(rule_base)
    >>> defuzzifiaction_method = karnik_mendel
    >>> mamdani.infer(defuzzifiaction_method, features)
    0.5
    """
    _rule_base: Iterable[Rule]

    def __init__(self, rule_base: Iterable[Rule]):
        """
        Create mamdani inference system with given rule base
        All rules should have the same consequent type and consequents should be defined on the same domain
        :param rule_base: fuzzy knowledge base used for inference
        """
        super().__init__(rule_base)
        self.__validate_consequents()

    def infer(self, defuzzification_method: Callable, features: Dict[Clause, List[MembershipDegree]]) \
            -> Sequence[float] or float:
        """
        Inferences output based on features of given object using chosen method
        :param defuzzification_method: 'KM', 'COG', 'LOM', 'MOM', 'SOM', 'MeOM', 'COS'
        :param features: dictionary of linguistic variables and their values
        :return: decision value
        """
        if not isinstance(features, Dict):
            raise ValueError("Features must be fuzzified dictionary")
        if not isinstance(defuzzification_method, Callable):
            raise ValueError("Defuzzifiaction method must be callable")

        degrees = self.__get_degrees(features)
        is_type1 = self.__is_consequent_type1()
        result = np.zeros(shape=(1, degrees))

        for i in range(degrees):
            single_features = {}
            for clause, memberships in features.items():
                single_features[clause] = np.take(memberships, i, axis=-1)

            if is_type1:
                domain, membership_functions = self.__get_domain_and_consequents_membership_functions(single_features)
                result[i] = defuzzification_method(domain, membership_functions)
            else:
                domain, lmfs, umfs = self.__get_domain_and_consequents_memberships_for_it2(single_features)
                result[:, i] = defuzzification_method(lmfs, umfs, domain)

        if result.shape[1] == 1:
            return result.item()

        return np.squeeze(result, 0)

    def __get_degrees(self, features: Dict[Clause, List[MembershipDegree]]) -> List[MembershipDegree]:
        values = np.array(list(features.values()))
        return values[0].shape[1]

    def __validate_consequents(self) -> NoReturn:
        for rule in self._rule_base:
            if not isinstance(rule.consequent, MamdaniConsequent):
                raise ValueError("All rule consequents must be mamdani consequents")

    def __is_consequent_type1(self) -> bool:
        return isinstance(self._rule_base[0].consequent.clause.fuzzy_set, Type1FuzzySet)

    def __get_domain_and_consequents_memberships_for_it2(self, features: Dict[Clause, List[MembershipDegree]]) \
            -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Extracts domain and membership functions from rule base
        :param features: dictionary of linguistic variables and their values
        :return: domain, lower membership functions and upper membership functions extracted from rule base
        """
        domain, membership_functions = self.__get_domain_and_consequents_membership_functions(features)
        lmfs = [membership_function[0] for membership_function in membership_functions]
        umfs = [membership_function[1] for membership_function in membership_functions]
        return domain, lmfs, umfs

    def __get_domain_and_consequents_membership_functions(self, features: Dict[Clause, List[MembershipDegree]]) \
            -> Tuple[np.ndarray, List[np.ndarray]]:
        domain = self.__get_consequent_domain()
        membership_functions = self.__get_consequents_membership_functions(features)
        return domain, membership_functions

    def __get_consequents_membership_functions(self, features: Dict[Clause, List[MembershipDegree]]) -> np.ndarray:
        """
        Extracts rule outputs from rule base
        :param features: dictionary of linguistic variables and their values
        :return: cut membership functions from rule base
        """
        return np.array([rule.consequent.output(rule.antecedent.fire(features)).values for rule in self._rule_base])

    def __get_consequent_domain(self) -> np.ndarray:
        return self._rule_base[0].consequent.clause.linguistic_variable.domain()

    @property
    def rule_base(self) -> Iterable[Rule]:
        return self._rule_base

    @rule_base.setter
    def rule_base(self, rule_base: Iterable[Rule]) -> NoReturn:
        if not isinstance(rule_base, abc.Iterable) or any(not isinstance(rule, Rule) for rule in rule_base):
            raise TypeError('rule_base must be an iterable of type Rule')
        self._rule_base = rule_base
