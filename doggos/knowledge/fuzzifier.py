import pandas as pd
import numpy as np
from typing import Iterable, Dict

from doggos.knowledge import Clause


def fuzzify(dataset: pd.DataFrame, clauses: Iterable[Clause]) -> Dict[Clause, np.ndarray]:
    """
    A function that takes a list of clauses and dataset, and calculate membership degrees to datasets in clause
    for each feature.
    :param dataset: input dataset
    :param clauses: list (or any other iterable structure) of clauses
    :return: dictionary where keys are clauses and items are lists of membership degrees to clause's fuzzy set
    for each feature

    Example:
    >>> import pandas as pd
    >>> from doggos.knowledge import Clause, LinguisticVariable, Domain
    >>> from doggos.fuzzy_sets import Type1FuzzySet
    >>> df = pd.DataFrame({'fire': [1.0, 2.3], 'air': [0, 2]})
    >>> df
       fire  air
    0   1.0    0
    1   2.3    2
    >>> lingustic_variable_1 = LinguisticVariable('fire', Domain(0, 5, 0.1))
    >>> lingustic_variable_2 = LinguisticVariable('air', Domain(0, 5, 0.1))
    >>> clause1 = Clause(lingustic_variable_1, 'high', Type1FuzzySet(lambda x: 1))
    >>> clause2 = Clause(lingustic_variable_1, 'low', Type1FuzzySet(lambda x: 0.8))
    >>> clause3 = Clause(lingustic_variable_2, 'high', Type1FuzzySet(lambda x: 0.9))
    >>> clause4 = Clause(lingustic_variable_2, 'low', Type1FuzzySet(lambda x: 0.7))
    >>> fuzzified = fuzzify(df, [clause1, clause2, clause3, clause4])
    >>> for key, item in fuzzified.items():
    ...     print(f'{key}: {item}')
    ...
    Clause fire is high: [1, 1]
    Clause fire is low: [0.8, 0.8]
    Clause air is high: [0.9, 0.9]
    Clause air is low: [0.7, 0.7]
    """
    fuzzified_dataset = dict()
    for clause in clauses:
        category = clause.linguistic_variable.name
        data = dataset[category].values
        fuzzified_dataset[clause] = clause.get_value(data)
    return fuzzified_dataset
