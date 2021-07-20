import copy
from typing import List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

from FuzzySetsParams import FuzzySetsParams
from doggos.algebras import LukasiewiczAlgebra, GodelAlgebra
from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet
from doggos.induction import FuzzyDecisionTableGenerator
from doggos.induction.rule_induction_WIP.inconsistencies_remover import InconsistenciesRemover
from doggos.induction.rule_induction_WIP.reductor import Reductor
from doggos.induction.rule_induction_WIP.rule_builder import RuleBuilder
from doggos.inference.takagi_sugeno_inference_system import TakagiSugenoInferenceSystem
from doggos.inference.defuzzification_algorithms import takagi_sugeno_karnik_mendel, weighted_average, \
    takagi_sugeno_EIASC
from doggos.knowledge import LinguisticVariable, Domain, Rule, fuzzify, Term, Clause
from doggos.knowledge.consequents import TakagiSugenoConsequent
from doggos.utils.membership_functions.membership_functions import generate_equal_gausses, gaussian, inv_gaussian_left,\
    inv_gaussian_right
import time
from pyswarm import pso


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


def main():
    best_accuracy = 0
    best_params = []
    final_rules = []

    # read dataset and normalize it
    df = pd.read_csv('data/Haberman.csv', sep=';')
    df_ar = df.values
    min_max_scaler = MinMaxScaler()
    df_scaled = min_max_scaler.fit_transform(df_ar)
    df = pd.DataFrame(df_scaled, columns=df.columns)
    df_y = df['Decision']




    rules = []
    clauses = []


    decision = LinguisticVariable('Decision', Domain(0, 1, 10))

    train, test = train_test_split(df, stratify=df['Decision'], test_size=0.3)
    train_y = train['Decision']
    classify_func = classify(0)

    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=23)
    fold = 0
    for train_index, test_index in skf.split(train, train_y):
        best_fold_params = []
        best_fold_acc = 0
        fold += 1
        print(f"Fold: {fold}")
        train_data = train.iloc[train_index]
        train_data_for_inference = train.iloc[train_index]
        train_y = train_data['Decision']

        fuzzy_params = FuzzySetsParams(train)
        mean_gausses_type1 = fuzzy_params.generate_5_t1_sets(["vsmall", "small", "medium", "large", "vlarge"])
        mean_gausses_type2 = fuzzy_params.generate_5_t2_sets(["vsmall", "small", "medium", "large", "vlarge"], 0.03)

        # generate fuzzy decision table
        gen = FuzzyDecisionTableGenerator(mean_gausses_type1, train_data)
        fuzzified_dataset = gen.fuzzify()

        # remove inconsistencies
        inc_rem = InconsistenciesRemover(fuzzified_dataset, list(fuzzified_dataset.columns)[:-1])
        decision_table, changed_decisions = inc_rem.inconsistenciesRemoving()

        # create reduct
        reductor = Reductor(decision_table, False)
        decision_table_with_reduct, features_number_after_reduct = reductor.worker(decision_table)

        # induce rules from DT created with type-1 FSs,
        # IT2 fuzzy sets are passed to fill antecedents with object
        rb = RuleBuilder(decision_table_with_reduct)
        antecedents, string_antecedents = rb.induce_rules(mean_gausses_type2)

        # define linguistic variables, get them from rule induction process
        ling_vars = list(rb.features)

        # define consequents and rules
        parameters_1 = {ling_vars[0]: -1.5, ling_vars[1]: -2.0}
        parameters_2 = {ling_vars[0]: 3.2, ling_vars[1]: 1.2}
        consequent_1 = TakagiSugenoConsequent(parameters_1, -1, decision)
        consequent_2 = TakagiSugenoConsequent(parameters_2, 1., decision)

        if (0.0 in antecedents and 1.0 in antecedents):
          rules = [Rule(antecedents[0.0], consequent_1), Rule(antecedents[1.0], consequent_2)]

        # use clauses generated in rule induction for fuzzyfing
        clauses = rb.clauses
        df_fuzzified = fuzzify(train_data_for_inference, clauses)

        # define measures
        data = train_data.reset_index().to_dict(orient='list')
        data.pop('index', None)

        measures = {ling_vars[0]: data['F0'],
                    ling_vars[1]: data['F1']}

        fitness = lambda parameters: evaluate(parameters, rules, ling_vars, df_fuzzified,
                                              measures, decision, classify_func, train_y)

        lb = [-200.] * 8
        ub = [200.] * 8

        #print('fitness')
        #fitness([-1.5, -2., -3.5, -5.5, 3.2, 1.2, 5.3, 2.4, -1, 1])
        xopt, fopt = pso(fitness, lb, ub, debug=True, maxiter=30, swarmsize=30)

        if (1 - fopt) > best_fold_acc:
            print(f"New best fold params {best_params} with accuracy {1 - fopt}!")
            best_fold_params = xopt
            best_fold_acc = 1 - fopt

        #best_fold_params = [1] * 6
        #validate on fold test data
        fold_test = train.iloc[test_index]
        fold_test_fuzzified = fuzzify(fold_test, clauses)

        fold_test_data = fold_test.reset_index().to_dict(orient='list')
        fold_test_data.pop('index', None)
        fold_test_measures = {}
        for feature in list(fold_test.columns)[:-1]:
            fold_test_measures[LinguisticVariable(str(feature), Domain(0, 1.001, 0.001))] = \
                fold_test_data[feature]

        print(f"Testing fold {fold} after PSO algorithm:")
        fold_test_result = evaluate_final(best_fold_params, rules, list(fold_test_measures.keys()), fold_test_fuzzified,
                                    fold_test_measures,
                                    decision, classify_func, fold_test['Decision'])

        if (1 - fold_test_result) > best_accuracy:
            print(f"New best accuracy found in fold: {fold}")
            best_accuracy = 1 - fold_test_result
            best_params = best_fold_params
            final_rules = copy.deepcopy(string_antecedents)




    # define measures
    data_test = test.reset_index().to_dict(orient='list')
    data_test.pop('index', None)
    ling_variables = []
    test_measures = {}
    for feature in list(test)[:-1]:
        ling_variables.append(LinguisticVariable(str(feature), Domain(0, 1.001, 0.001)))

    fuzzy_params = FuzzySetsParams(train)
    train_mean_gausses_type2 = fuzzy_params.generate_5_t2_sets(["vsmall", "small", "medium", "large", "vlarge"], 0.03)
    clauses, terms = return_clauses_and_terms(ling_variables, train_mean_gausses_type2)

    # validate on final test data after all folds
    test_fuzzified = fuzzify(test, clauses)

    rule1 = Rule(eval(final_rules[0], terms), rules[0].consequent)
    rule2 = Rule(eval(final_rules[1], terms), rules[1].consequent)


    for lv in ling_variables:
        test_measures[lv] = data_test[lv.name]

    print('ACCURACY ON FINAL TEST:')
    evaluate_final(best_params, [rule1, rule2], ling_variables, test_fuzzified, test_measures, decision, classify_func,
             test['Decision'])


def evaluate(params, rules_f: List[Rule], ling_variables, dataset, measures, decision, classify_func, y):
    f_params1 = {ling_variables[0]: params[0], ling_variables[1]: params[1]}
    f_params2 = {ling_variables[0]: params[2], ling_variables[1]: params[3]}
    print(params)
    rules_f[0].consequent.function_parameters = f_params1
    rules_f[1].consequent.function_parameters = f_params2
    rules_f[0].consequent.bias = params[4]
    rules_f[1].consequent.bias = params[5]
    ts = TakagiSugenoInferenceSystem(rules_f)
    result_eval = ts.infer(takagi_sugeno_EIASC, dataset, measures)
    y_pred_eval = list(map(lambda x: classify_func(x), result_eval[decision]))
    # print(y_pred)
    # print(df_y.values)
    accuracy1 = accuracy_score(y.values, y_pred_eval)

    print(f'Accuracy: {accuracy1:.5f}')
    return 1 - accuracy1

def evaluate_final(params, rules_f: List[Rule], ling_variables, dataset, measures, decision, classify_func, y):
    f_params1 = {ling_variables[0]: params[0], ling_variables[1]: params[1], ling_variables[2]: params[2]}
    f_params2 = {ling_variables[0]: params[3], ling_variables[1]: params[4], ling_variables[2]: params[5]}
    print(params)
    rules_f[0].consequent.function_parameters = f_params1
    rules_f[1].consequent.function_parameters = f_params2
    rules_f[0].consequent.bias = params[6]
    rules_f[1].consequent.bias = params[7]
    ts = TakagiSugenoInferenceSystem(rules_f)
    result_eval = ts.infer(takagi_sugeno_EIASC, dataset, measures)
    y_pred_eval = list(map(lambda x: classify_func(x), result_eval[decision]))
    # print(y_pred)
    # print(df_y.values)
    accuracy = accuracy_score(y.values, y_pred_eval)
    print("Test report", classification_report(y.values, y_pred_eval))
    print(f'Accuracy: {accuracy:.5f}')
    return 1 - accuracy


def return_clauses_and_terms(features, fuzzy_sets):
    algebra = GodelAlgebra()
    terms = {}
    clauses = []
    for feature in features:
        for key in fuzzy_sets[feature.name]:
            clause = Clause(feature, key, fuzzy_sets[feature.name][key])
            terms[f"{feature.name}_{key}"] = Term(algebra, clause)
            clauses.append(clause)
    return clauses, terms

if __name__ == '__main__':
    main()
