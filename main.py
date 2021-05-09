from os import path
import sys
from typing import List, Dict

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from doggos.algebras import LukasiewiczAlgebra, GodelAlgebra
from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet
from doggos.induction import FuzzyDecisionTableGenerator
from doggos.induction.rule_induction_WIP.inconsistencies_remover import InconsistenciesRemover
from doggos.induction.rule_induction_WIP.reductor import Reductor
from doggos.induction.rule_induction_WIP.rule_builder import RuleBuilder
from doggos.inference.takagi_sugeno_inference_system import TakagiSugenoInferenceSystem
from doggos.inference.defuzzification_algorithms import takagi_sugeno_karnik_mendel, weighted_average, \
    takagi_sugeno_EIASC
from doggos.knowledge import LinguisticVariable, Domain, Rule, fuzzify, Term
from doggos.knowledge.consequents import TakagiSugenoConsequent
from doggos.utils.membership_functions.membership_functions import generate_equal_gausses, gaussian
import time
from pyswarm import pso

start_time = time.time()
print("Start:")
# define fuzzy sets and save into dict
gausses = generate_equal_gausses(3, 0, 1)
gausses = [gaussian(.0, 0.21), gaussian(.5, 0.21), gaussian(1., 0.21)]
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
# print(fuzzified_dataset)

# remove inconsistencies
inc_rem = InconsistenciesRemover(fuzzified_dataset, list(fuzzified_dataset.columns)[:-1])
decision_table, changed_decisions = inc_rem.inconsistenciesRemoving()
# print(decision_table)
# print(changed_decisions)

# create reduct
reductor = Reductor(decision_table, True)
decision_table_with_reduct, features_number_after_reduct = reductor.worker(decision_table)
# print(decision_table_with_reduct)

# define IT2FSs
gausses_LMF = [gaussian(.0, 0.20), gaussian(.5, 0.20), gaussian(1., 0.20)]
gausses_UMF = [gaussian(.0, 0.22), gaussian(.5, 0.22), gaussian(1., 0.22)]

small_T2 = IntervalType2FuzzySet(gausses_LMF[0], gausses_UMF[0])
medium_T2 = IntervalType2FuzzySet(gausses_LMF[1], gausses_UMF[1])
large_T2 = IntervalType2FuzzySet(gausses_LMF[2], gausses_UMF[2])
fuzzy_sets_T2 = {'small': small_T2, 'medium': medium_T2, 'large': large_T2}

# induce rules
rb = RuleBuilder(decision_table_with_reduct)
antecedents = rb.induce_rules(fuzzy_sets_T2)
# print(antecedents)
print("After rule induction: --- %s seconds ---" % (time.time() - start_time))

# define linguistic variables, get them from rule induction process
decision = LinguisticVariable('Decision', Domain(0, 1, 0.001))
ling_vars = list(rb.features)
# print(ling_vars)

# define consequents and rules
parameters_1 = {ling_vars[0]: -1.5, ling_vars[1]: -2.0, ling_vars[2]: -3.5, ling_vars[3]: -5.5}
parameters_2 = {ling_vars[0]: 3.2, ling_vars[1]: 1.2, ling_vars[2]: 5.3, ling_vars[3]: 2.4}
consequent_1 = TakagiSugenoConsequent(parameters_1, -1, decision)
consequent_2 = TakagiSugenoConsequent(parameters_2, 1., decision)

rules = [Rule(antecedents[0.0], consequent_1), Rule(antecedents[1.0], consequent_2)]

# use clauses generated in rule induction for fuzzyfying
clauses = rb.clauses
df_fuzzified = fuzzify(df, clauses)
# print(df_fuzzified)


# define measures
data = df.reset_index().to_dict(orient='list')
data.pop('index', None)

measures = {ling_vars[0]: data['F0'],
            ling_vars[1]: data['F1'],
            ling_vars[2]: data['F2'],
            ling_vars[3]: data['F3']}

# infer
inference_system = TakagiSugenoInferenceSystem(rules)
result = inference_system.infer(takagi_sugeno_EIASC, df_fuzzified, measures)
# print(result)

theta = 0


# classification function
def classify(theta):
    def _classify(x):
        if x <= theta:
            return 0
        elif x > theta:
            return 1
        else:
            print('else')

    return _classify


classify_func = classify(theta)
y_pred = list(map(lambda x: classify_func(x), result[decision]))
# print(y_pred)
# print(df_y.values)
accuracy = accuracy_score(df_y.values, y_pred)
print(f'Accuracy: {accuracy:.5f}')
print("After inference : --- %s seconds ---" % (time.time() - start_time))


def evaluate(params, rules_f: List[Rule], ling_variables, dataset):
    f_params1 = {ling_variables[0]: params[0], ling_variables[1]: params[1], ling_variables[2]: params[2],
                 ling_variables[3]: params[3]}
    f_params2 = {ling_variables[0]: params[4], ling_variables[1]: params[5], ling_variables[2]: params[6],
                 ling_variables[3]: params[7]}
    print(params)
    rules_f[0].consequent.function_parameters = f_params1
    rules_f[1].consequent.function_parameters = f_params2
    rules_f[0].consequent.bias = params[8]
    rules_f[1].consequent.bias = params[9]
    ts = TakagiSugenoInferenceSystem(rules_f)
    result_eval = ts.infer(takagi_sugeno_EIASC, dataset, measures)
    y_pred_eval = list(map(lambda x: classify_func(x), result_eval[decision]))
    #print(y_pred)
    #print(df_y.values)
    accuracy1 = accuracy_score(df_y.values, y_pred_eval)

    print(f'Accuracy: {accuracy1:.5f}')
    return 1 - accuracy1


fitness = lambda parameters: evaluate(parameters, rules, ling_vars, df_fuzzified)
#fitness([[1., 1., 1., 1.], [-1., -1., -1., -1.], -2., 2.])
print("After inference again : --- %s seconds ---" % (time.time() - start_time))
lb = [-80., -80., -80., -80., -20., -20., -20., -20., -80., -20.]
ub = [20., 20., 20., 20., 80., 80., 80., 80., 20., 80.]

print('ftiness')
fitness([-1.5, -2., -3.5, -5.5, 3.2, 1.2, 5.3, 2.4, -1, 1])
xopt, fopt = pso(fitness, lb, ub, debug=True)