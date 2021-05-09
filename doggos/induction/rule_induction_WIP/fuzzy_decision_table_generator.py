import pandas as pd
from typing import Dict
from doggos.fuzzy_sets.type1_fuzzy_set import Type1FuzzySet
from doggos.knowledge import Clause, LinguisticVariable, Domain


class FuzzyDecisionTableGenerator:

    def __init__(self, fuzzy_sets: Dict[str, Type1FuzzySet], dataset: pd.DataFrame):
        self.__fuzzy_sets = fuzzy_sets
        self.__dataset = dataset
        self.__features = []
        for feature in dataset.columns:
            self.__features.append(LinguisticVariable(str(feature), Domain(0, 1.001, 0.001)))
        self.__features_clauses = {col: [] for col in list(dataset.columns)}

    def get_highest_membership(self, feature: str, input: float):
        max_feature = None
        max_value = 0
        for clause in self.__features_clauses[feature]:
            if clause.get_value(input) > max_value:
                max_feature = clause.gradation_adjective
                max_value = clause.get_value(input)
        return max_feature

    def fuzzify(self):
        for feature in self.__features:
            self.__features_clauses[feature] = []
            for key in self.__fuzzy_sets:
                self.__features_clauses[feature.name].append(Clause(feature, key, self.__fuzzy_sets[key]))

        fuzzy_dataset = pd.DataFrame(list([self.__dataset.columns]), dtype="string")
        fuzzy_dataset.columns = self.__dataset.columns
        fuzzy_dataset.astype('str')
        fuzzy_dataset["Decision"] = pd.to_numeric(fuzzy_dataset["Decision"], errors='ignore')
        for i, row in self.__dataset.iterrows():
            for f in self.__dataset:
                if f == 'Decision':
                    var = self.__dataset.at[i, f]
                    fuzzy_dataset.at[i, f] = var
                else:
                    fuzzy_dataset.at[i, f] = self.get_highest_membership(f, self.__dataset.at[i, f])

        return fuzzy_dataset

