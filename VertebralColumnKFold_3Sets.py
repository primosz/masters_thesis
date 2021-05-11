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
from doggos.utils.membership_functions.membership_functions import generate_equal_gausses, gaussian
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
    final_clauses = []
    final_measures = {}

    # read dataset and normalize it
    df = pd.read_csv('data\Vertebral.csv', sep=';')
    df_ar = df.values
    min_max_scaler = MinMaxScaler()
    df_scaled = min_max_scaler.fit_transform(df_ar)
    df = pd.DataFrame(df_scaled, columns=df.columns)
    df_y = df['Decision']


    rules = []
    clauses = []

    decision = LinguisticVariable('Decision', Domain(0, 1, 10))

    train, test = train_test_split(df, stratify=df['Decision'], test_size=0.2)
    train_y = train['Decision']
    classify_func = classify(0)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=23)
    fold = 0
    for train_index, test_index in skf.split(train, train_y):
        best_fold_params = []
        best_fold_acc = 0
        fold += 1
        print(f"Fold: {fold}")
        train_data = train.iloc[train_index]
        train_data_for_inference = train.iloc[train_index]
        train_y = train_data['Decision']

        fuzzy_params = FuzzySetsParams(train_data)
        mean_gausses_type1 = fuzzy_params.generate_3_t1_sets(["small", "medium", "large"])
        mean_gausses_type2 = fuzzy_params.generate_3_t2_sets(["small", "medium", "large"], 0.01)

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
        parameters_1 = {ling_vars[0]: -1.5, ling_vars[1]: -2.0, ling_vars[2]: -3.5,
                        ling_vars[3]: -5.5, ling_vars[4]: -5.5, ling_vars[5]: -1.5}
        parameters_2 = {ling_vars[0]: 1.5, ling_vars[1]: 2.0, ling_vars[2]: 3.5,
                        ling_vars[3]: 5.5, ling_vars[4]: 5.5, ling_vars[5]: 1.5}
        consequent_1 = TakagiSugenoConsequent(parameters_1, -1, decision)
        consequent_2 = TakagiSugenoConsequent(parameters_2, 1., decision)

        rules = [Rule(antecedents[0.0], consequent_1), Rule(antecedents[1.0], consequent_2)]

        # use clauses generated in rule induction for fuzzyfing
        clauses = rb.clauses
        df_fuzzified = fuzzify(train_data_for_inference, clauses)

        # define measures
        data = train_data.reset_index().to_dict(orient='list')
        data.pop('index', None)

        measures = {ling_vars[0]: data['F0'],
                    ling_vars[1]: data['F1'],
                    ling_vars[2]: data['F2'],
                    ling_vars[3]: data['F3'],
                    ling_vars[4]: data['F4'],
                    ling_vars[5]: data['F5']
                    }

        fitness = lambda parameters: evaluate(parameters, rules, ling_vars, df_fuzzified,
                                              measures, decision, classify_func, train_y)

        lb = [-80.] * 14
        ub = [80.] * 14

        #print('fitness')
        #fitness([-1.5, -2., -3.5, -5.5, 3.2, 1.2, 5.3, 2.4, -1, 1])
        xopt, fopt = pso(fitness, lb, ub, debug=True, maxiter=40, swarmsize=40)

        if (1 - fopt) > best_fold_acc:
            print(f"New best fold params {best_params} with accuracy {1 - fopt}!")
            best_fold_params = xopt
            best_fold_acc = 1 - fopt

        #best_fold_params = [-1.5, -2., -3.5, -5.5, 3.2, 1.2, 5.3, 2.4, -1, 1]
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
        fold_test_result = evaluate(best_fold_params, rules, list(fold_test_measures.keys()), fold_test_fuzzified,
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
    train_mean_gausses_type2 = fuzzy_params.generate_3_t2_sets(["small", "medium", "large"], 0.01)
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


def evaluate(params, rules_f: List[Rule], lv, dataset, measures, decision, classify_func, y):
    f_params1 = {lv[0]: params[0], lv[1]: params[1], lv[2]: params[2],
                 lv[3]: params[3], lv[4]: params[4], lv[5]: params[5]}
    f_params2 = {lv[0]: params[6], lv[1]: params[7], lv[2]: params[8],
                 lv[3]: params[9], lv[4]: params[10], lv[5]: params[11]}
    print(params)
    rules_f[0].consequent.function_parameters = f_params1
    rules_f[1].consequent.function_parameters = f_params2
    rules_f[0].consequent.bias = params[12]
    rules_f[1].consequent.bias = params[13]
    ts = TakagiSugenoInferenceSystem(rules_f)
    result_eval = ts.infer(takagi_sugeno_EIASC, dataset, measures)
    y_pred_eval = list(map(lambda x: classify_func(x), result_eval[decision]))
    # print(y_pred)
    # print(df_y.values)
    accuracy1 = accuracy_score(y.values, y_pred_eval)

    print(f'Accuracy: {accuracy1:.5f}')
    return 1 - accuracy1


def evaluate_final(params, rules_f: List[Rule], lv, dataset, measures, decision, classify_func, y):
    f_params1 = {lv[0]: params[0], lv[1]: params[1], lv[2]: params[2],
                 lv[3]: params[3], lv[4]: params[4], lv[5]: params[5]}
    f_params2 = {lv[0]: params[6], lv[1]: params[7], lv[2]: params[8],
                 lv[3]: params[9], lv[4]: params[10], lv[5]: params[11]}
    print(params)
    rules_f[0].consequent.function_parameters = f_params1
    rules_f[1].consequent.function_parameters = f_params2
    rules_f[0].consequent.bias = params[12]
    rules_f[1].consequent.bias = params[13]
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
        for key in fuzzy_sets:
            clause = Clause(feature, key, fuzzy_sets[key])
            terms[f"{feature.name}_{key}"] = Term(algebra, clause)
            clauses.append(clause)
    return clauses, terms

if __name__ == '__main__':
    main()
