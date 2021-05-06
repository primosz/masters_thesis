from os import path
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet
from doggos.induction import FuzzyDecisionTableGenerator
from doggos.induction.rule_induction_WIP.inconsistencies_remover import InconsistenciesRemover
from doggos.induction.rule_induction_WIP.reductor import Reductor
from doggos.induction.rule_induction_WIP.rule_builder import RuleBuilder
from doggos.knowledge import LinguisticVariable, Domain, Rule, fuzzify
from doggos.knowledge.consequents import TakagiSugenoConsequent
from doggos.utils.membership_functions.membership_functions import generate_equal_gausses, gaussian

sys.path.append((path.abspath('../biblioteka/DoggOSFuzzy/doggos')))

#define fuzzy sets and save into dict
gausses = generate_equal_gausses(3, 0, 1)
small = Type1FuzzySet(gausses[0])
medium = Type1FuzzySet(gausses[1])
large = Type1FuzzySet(gausses[2])

fuzzy_sets = {'small': small, 'medium': medium, 'large': large}

#read dataset and normalize it
df = pd.read_csv('Data Banknote Authentication.csv', sep=';')
df_ar = df.values
min_max_scaler = MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df_ar)
df = pd.DataFrame(df_scaled, columns=df.columns)

#fuzzify dataset
gen = FuzzyDecisionTableGenerator(fuzzy_sets, df)
fuzzified_dataset = gen.fuzzify()
print(fuzzified_dataset)

#remove inconsistencies
inc_rem = InconsistenciesRemover(fuzzified_dataset, list(fuzzified_dataset.columns)[:-1])
decision_table, changed_decisions = inc_rem.inconsistenciesRemoving()
print(decision_table)
print(changed_decisions)

#create reduct
reductor = Reductor(decision_table, True)
decision_table_with_reduct, features_number_after_reduct = reductor.worker(decision_table)
print(decision_table_with_reduct)


gausses_LMF = [gaussian(.0, .20), gaussian(.5, .20), gaussian(1., .20)]
gausses_UMF = [gaussian(.0, .22), gaussian(.5, .22), gaussian(1., .22)]

small_T2 = IntervalType2FuzzySet(gausses_LMF[0], gausses_UMF[0])
medium_T2 = IntervalType2FuzzySet(gausses_LMF[0], gausses_UMF[0])
large_T2 = IntervalType2FuzzySet(gausses_LMF[0], gausses_UMF[0])

fuzzy_sets_T2 = {'small': small, 'medium': medium, 'large': large}
#induce rules
rb = RuleBuilder(decision_table_with_reduct)
antecedents= rb.induce_rules(fuzzy_sets_T2)
print(antecedents)

decision = LinguisticVariable('Decision', Domain(0, 1, 0.001))
ling_vars = RuleBuilder.features

parameters_1 = {ling_vars[0]: 0.25, RuleBuilder.features[1]: 0.25, RuleBuilder.features[2]: 0.5, RuleBuilder.features[3]: 0.5}
parameters_2 = {RuleBuilder.features[0]: 0.75, RuleBuilder.features[1]: 0.65, RuleBuilder.features[2]: 0.85, RuleBuilder.features[3]: 0.55}
consequent_1 = TakagiSugenoConsequent(parameters_1, 0, decision)
consequent_2 = TakagiSugenoConsequent(parameters_2, 0, decision)

rules = [Rule(antecedents["0.0"], consequent_1), Rule(antecedents["1.0"], consequent_2)]

clauses = rb.return_clauses(fuzzy_sets_T2)
df_fuzzified = fuzzify(df, clauses)

#map string rules to fuzzy rules with type-2 fuzzy sets
