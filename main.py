from os import path
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from doggos.fuzzy_sets import Type1FuzzySet
from doggos.induction import FuzzyDecisionTableGenerator
from doggos.induction.rule_induction_WIP.inconsistencies_remover import InconsistenciesRemover
from doggos.induction.rule_induction_WIP.reductor import Reductor
from doggos.induction.rule_induction_WIP.rule_builder import RuleBuilder
from doggos.utils.membership_functions.membership_functions import generate_equal_gausses

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

#induce rules
rb = RuleBuilder(decision_table_with_reduct)
rules = rb.induce_rules()
print(rules)

#map string rules to fuzzy rules with type-2 fuzzy sets
