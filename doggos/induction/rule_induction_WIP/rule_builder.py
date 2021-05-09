import pandas as pd

from doggos.algebras import GodelAlgebra
from doggos.knowledge import Clause, LinguisticVariable, Domain, Term


class RuleBuilder:

    def __init__(self, dataset: pd.DataFrame):

        self.__decision_rules = {}
        self.__decisions = None
        self.__dataset = dataset
        self.__features = []
        columns = list(dataset.columns)
        columns.remove('Decision')
        for feature in columns:
            self.__features.append(LinguisticVariable(str(feature), Domain(0, 1.001, 0.001)))
        self.__terms = {}
        self.__clauses = []

    def induce_rules(self, fuzzy_sets):
        algebra = GodelAlgebra()
        for feature in self.__features:
            for key in fuzzy_sets:
                clause = Clause(feature, key, fuzzy_sets[key])
                self.__terms[f"{feature.name}_{key}"] = Term(algebra, clause)
                self.__clauses.append(clause)
        differences = self.get_differences(self.__dataset)
        classes = set(self.__dataset['Decision'])
        decision_records = dict([(clazz, []) for clazz in classes])
        for i, row in self.__dataset.iterrows():
            decision_records[row['Decision']].append((row, i))

        self.__decisions = []
        decision_rules = {}
        for clazz in decision_records:
            self.__decisions.append(clazz)
            row = decision_records[clazz]
            decision_rules[clazz] = self.build_rule(differences, row)
        self.__decision_rules = decision_rules
        return self.__decision_rules

    def build_rule(self, differences, records):
        all_conjunction = []
        for r, i in records:
            conjunction = self.get_implicants(differences, r, i)
            if len(conjunction) > 0:
                all_conjunction.append(conjunction)
        all_conjunction.sort(key=lambda x: len(x))
        res_conjunction = []
        for con in all_conjunction:
            res_conjunction.append(sorted(con, key=lambda x: len(x)))
        if len(res_conjunction) == 0:
            antecedent = None
        else:
            antecedent = ""
            for ai, a in enumerate(res_conjunction):
                if ai != 0:
                    antecedent += " | "
                antecedent += "("
                for bi, b in enumerate(a):
                    if bi != 0:
                        antecedent += " & "
                    antecedent += "("
                    for ci, c in enumerate(b):
                        if ci != 0:
                            antecedent += " | "
                        antecedent += c
                    antecedent += ")"
                antecedent += ")"
        return eval(antecedent, self.__terms)

    def get_implicants(self, differences, record, index):

        diff_copy = []
        for df in differences[index]:
            if df not in diff_copy:
                diff_copy.append(df)
        diff_copy = sorted(diff_copy)
        diff_copy = sorted(diff_copy, key=lambda x: 1 / len(x))
        all_alternatives = []
        for diff in diff_copy:
            alternative = None
            for a in diff:
                if alternative is None:
                    alternative = [a + "_" + record[a]]
                elif a + "_" + record[a] not in alternative:
                    alternative.append(a + "_" + record[a])
            all_alternatives.append(alternative)
        all_alternatives.sort(key=lambda x: len(x))
        res_alternatives = []
        for alt in all_alternatives:
            res_alternatives.append(sorted(alt))

        return res_alternatives

    def get_differences(self, dataset):
        differences = dict()
        for i, row in dataset.iterrows():
            first = row
            differences[i] = []
            for j, row2 in dataset.iterrows():
                second = row2
                if i == j:
                    continue
                difference = [attr for attr in dataset if 'Decision' not in attr and first[attr] != second[attr]]
                if len(differences) != 0:
                    differences[i].append(difference)
        return differences

    @property
    def clauses(self):
        return self.__clauses

    @property
    def features(self):
        return self.__features

    @features.setter
    def features(self, features):
        self.__features = features

