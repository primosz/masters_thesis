from os import path
import sys
from typing import List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from pyswarm import pso

from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet
from doggos.induction import FuzzyDecisionTableGenerator
from doggos.induction.rule_induction_WIP.inconsistencies_remover import InconsistenciesRemover
from doggos.induction.rule_induction_WIP.reductor import Reductor
from doggos.induction.rule_induction_WIP.rule_builder import RuleBuilder
from doggos.inference import TakagiSugenoInferenceSystem
from doggos.inference.defuzzification_algorithms import takagi_sugeno_karnik_mendel
from doggos.knowledge import LinguisticVariable, Domain, Rule, fuzzify
from doggos.knowledge.consequents import TakagiSugenoConsequent
from doggos.utils.membership_functions.membership_functions import generate_equal_gausses, gaussian

sys.path.append((path.abspath('../biblioteka/DoggOSFuzzy/doggos')))

# define fuzzy sets and save into dict
gausses = generate_equal_gausses(3, 0, 1)
small = Type1FuzzySet(gausses[0])
medium = Type1FuzzySet(gausses[1])
large = Type1FuzzySet(gausses[2])

fuzzy_sets = {'small': small, 'medium': medium, 'large': large}

# read dataset and normalize it
df = pd.read_csv('Data Banknote Authentication.csv', sep=';')
df_ar = df.values
min_max_scaler = MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df_ar)
df = pd.DataFrame(df_scaled, columns=df.columns)
df_y = df['Decision']

# fuzzify dataset
gen = FuzzyDecisionTableGenerator(fuzzy_sets, df)
fuzzified_dataset = gen.fuzzify()
print(fuzzified_dataset)

# remove inconsistencies
inc_rem = InconsistenciesRemover(fuzzified_dataset, list(fuzzified_dataset.columns)[:-1])
decision_table, changed_decisions = inc_rem.inconsistenciesRemoving()
print(decision_table)
print(changed_decisions)

# create reduct
reductor = Reductor(decision_table, True)
decision_table_with_reduct, features_number_after_reduct = reductor.worker(decision_table)
print(decision_table_with_reduct)

gausses_LMF = [gaussian(.0, .20), gaussian(.5, .20), gaussian(1., .20)]
gausses_UMF = [gaussian(.0, .22), gaussian(.5, .22), gaussian(1., .22)]

small_T2 = IntervalType2FuzzySet(gausses_LMF[0], gausses_UMF[0])
medium_T2 = IntervalType2FuzzySet(gausses_LMF[0], gausses_UMF[0])
large_T2 = IntervalType2FuzzySet(gausses_LMF[0], gausses_UMF[0])

fuzzy_sets_T2 = {'small': small_T2, 'medium': medium_T2, 'large': large_T2}
# induce rules
rb = RuleBuilder(decision_table_with_reduct)
antecedents = rb.induce_rules(fuzzy_sets_T2)
print(antecedents)

decision = LinguisticVariable('Decision', Domain(0, 1, 0.001))
ling_vars = list(rb.features)
print(ling_vars)


parameters_1 = {ling_vars[0]: -1.1, ling_vars[1]: -1., ling_vars[2]: -1., ling_vars[3]: -1.}
parameters_2 = {ling_vars[0]: 2.2, ling_vars[1]: 2., ling_vars[2]: 20., ling_vars[3]: 2.}
consequent_1 = TakagiSugenoConsequent(parameters_1, 0, decision)
consequent_2 = TakagiSugenoConsequent(parameters_2, 0, decision)

rules = [Rule(antecedents[0.0], consequent_1), Rule(antecedents[1.0], consequent_2)]


clauses = rb.clauses
df_fuzzified = fuzzify(df, clauses)
print(df_fuzzified)

data = df.reset_index().to_dict(orient='list')
data.pop('index', None)

measures = {ling_vars[0]: data['F0'],
            ling_vars[1]: data['F1'],
            ling_vars[2]: data['F2'],
            ling_vars[3]: data['F3']}

inference_system = TakagiSugenoInferenceSystem(rules)
result = inference_system.infer(takagi_sugeno_karnik_mendel, df_fuzzified, measures)
print(result)

theta = 3.1


def classify(theta):
    def _classify(x):
        if x < theta:
            return 0
        elif x > theta:
            return 1
        else:
            return None
    return _classify


classify_func = classify(theta)
y_pred = list(map(lambda x: classify_func(x), result[decision]))
print(y_pred)
accuracy = accuracy_score(y_pred, df_y.values)
print(f'Accuracy: {accuracy:.5f}')

def evaluate(param1, param2, rules: List[Rule], dataset):
    rules[0].consequent.function_parameters = param1
    rules[1].consequent.function_parameters = param2
    #ts = TakagiSugenoInferenceSystem(rules)
    #result = inference_system.infer(takagi_sugeno_karnik_mendel, dataset, measures)
    print(dataset)

ff = lambda dataset: evaluate(parameters_1, parameters_2, rules, dataset)
ff("asd")



