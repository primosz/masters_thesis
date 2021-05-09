from collections.abc import Iterable
from functools import partial
from typing import List
import math
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats.mstats_basic import trim


def center_of_gravity(domain: np.ndarray, membership_functions: List[np.ndarray]) -> float:
    cut = __membership_func_union(membership_functions)
    return np.average(domain, weights=cut)


def largest_of_maximum(domain: np.ndarray, membership_functions: List[np.ndarray]) -> float:
    cut = __membership_func_union(membership_functions)
    maximum = np.max(cut)
    return domain[np.where(cut == maximum)[0][-1]]


def smallest_of_maximum(domain: np.ndarray, membership_functions: List[np.ndarray]) -> float:
    cut = __membership_func_union(membership_functions)
    maximum = np.max(cut)
    return domain[np.where(cut == maximum)[0][0]]


def middle_of_maximum(domain: np.ndarray, membership_functions: List[np.ndarray]) -> float:
    cut = __membership_func_union(membership_functions)
    maximum = np.max(cut)
    indices = np.where(cut == maximum)[0]
    size = indices.size
    middle = int(size / 2)
    return domain[[indices[middle]]]


def mean_of_maxima(domain: np.ndarray, membership_functions: List[np.ndarray]) -> float:
    cut = __membership_func_union(membership_functions)
    maximum = np.max(cut)
    indices = np.where(cut == maximum)[0]
    size = indices.size
    total = np.sum([domain[index] for index in indices])
    return total / size


def center_of_sums(domain: np.ndarray, membership_functions: List[np.ndarray]) -> float:
    if not isinstance(membership_functions[0], Iterable):
        sums_of_memberships = membership_functions
    else:
        universe = np.array(membership_functions)
        sums_of_memberships = np.sum(universe, axis=0)

    domain_memberships_sums = np.array((domain, sums_of_memberships))
    numerator = np.sum(np.prod(domain_memberships_sums, axis=0))
    denominator = np.sum(sums_of_memberships)

    return numerator / denominator


def karnik_mendel(lmfs: List[np.ndarray], umfs: List[np.ndarray], domain: np.ndarray) -> float:
    """
    Karnik-Mendel algorithm for interval type II fuzzy sets
    :param lmfs: lower membership functions
    :param umfs: upper membership functions
    :param domain: universe on which rule consequents are defined
    :return: decision value
    """
    lower_cut = __membership_func_union(lmfs)
    upper_cut = __membership_func_union(umfs)
    thetas = (lower_cut + upper_cut) / 2 + 1e-10
    y_l = __find_y(partial(__find_c_minute, under_k_mf=upper_cut, over_k_mf=lower_cut), domain, thetas)
    y_r = __find_y(partial(__find_c_minute, under_k_mf=lower_cut, over_k_mf=upper_cut), domain, thetas)
    return (y_l + y_r) / 2


def EIASC(intervals: np.ndarray):
    """
    EIASC algorithm

    Parameters
    ----------

    intervals:
        numpy (n, 4) array

        Y = intervals[:, 0:2]

        F = intervals[:, 2:4]


    Returns
    -------
    Centroid of interval: float
    """

    # Left calculations
    intervals = intervals[intervals[:, 0].argsort()]
    intervals = trim_intervals(intervals)

    if intervals is False:
        return 0, 0

    N = len(intervals)
    b = np.sum(intervals[:, 2])

    a_l = np.sum(intervals[:, 0] * intervals[:, 2])
    b_l = b
    L = 0
    while True:
        d = intervals[L, 3] - intervals[L, 2]
        a_l += intervals[L, 0] * d
        b_l += d
        y_l = a_l / b_l
        L += 1
        if (y_l <= intervals[L, 0]) or math.isclose(y_l, intervals[L, 0]):
            break
            # Right calculations
    intervals = intervals[intervals[:, 1].argsort()]
    a_r = np.sum(intervals[:, 1] * intervals[:, 2])
    b_r = b
    R = N - 1
    while True:
        d = intervals[R, 3] - intervals[R, 2]
        a_r += intervals[R, 1] * d
        b_r += d
        y_r = a_r / b_r
        R -= 1
        if (y_r >= intervals[R, 1]) or math.isclose(y_r, intervals[R, 1]):
            break
    return (y_l + y_r)/2


def trim_intervals(intervals):
    v = intervals[:, 3]
    i, = np.where(v > 0)
    if i.size == 0:
        return False
    else:
        min1 = i[0]
        max1 = i[-1] + 1

        v = intervals[:, 2]
        i, = np.where(v > 0)
        if i.size == 0:
            min2 = min1
            max2 = max1
        else:
            min2 = i[0]
            max2 = i[-1] + 1
        return intervals[min(min1, min2):max(max1, max2), :]


def __find_y(partial_find_c_minute: partial, domain: np.ndarray, thetas: np.ndarray) -> float:
    """
    Finds decision factor for specified part of algorithm
    :param partial_find_c_minute: _find_c_minute function with filled under_k_mf and over_k_mf arguments
    :param domain: universe on which rule consequents are defined
    :param thetas: weights for weighted average: (lmf + umf) / 2
    :return: Decision factor for specified part of algorithm
    """
    c_prim = np.average(domain, weights=thetas)
    c_minute = partial_find_c_minute(c=c_prim, domain=domain)
    while abs(c_minute - c_prim) > np.finfo(float).eps:
        c_prim = c_minute
        c_minute = partial_find_c_minute(c=c_prim, domain=domain)
    return c_minute


def __find_c_minute(c: float, under_k_mf: np.ndarray, over_k_mf: np.ndarray,
                    domain: np.ndarray) -> float:
    """
    Finds weights and average for combined membership functions
    :param c: weighted average of domain values with previously defined thetas as weights
    :param under_k_mf: takes elements of under_k_mf with indices <= k as weights
    :param over_k_mf: takes elements of over_k_mf with indices >= k+1 as weights
    :param domain: universe on which rule consequents are defined
    :return: average for combined membership functions
    """
    k = __find_k(c, domain)
    weights = np.zeros(shape=domain.size)
    weights[:(k + 1)] = under_k_mf[:(k + 1)]
    weights[(k + 1):] = over_k_mf[(k + 1):]
    weights += 1e-10
    return np.average(domain, weights=weights)


def __find_k(c: float, domain: np.ndarray) -> float:
    """
    Finds index for weighted average in given domain
    :param c: weighted average of combined membership functions
    :param domain: universe on which rule consequents are defined
    :return: index for weighted average in given domain
    """
    return np.where(domain <= c)[0][-1]


def weighted_average(firings: np.ndarray,
                     outputs: np.ndarray) -> float:
    """
    Method of calculating output of Takagi-Sugeno inference system for fuzzy sets of type 1
    Used for fuzzy sets of type 1.

    :param firings: firings of rules
    :param outputs: outputs of rules
    :return: float that is weighted average of all outputs with firings as their weights
    """
    return np.average(outputs, weights=firings + 1e-10)


def takagi_sugeno_karnik_mendel(firings: np.ndarray,
                                outputs: np.ndarray,
                                step: float = 0.01) -> float:
    """
    Method of calculating output of Takagi-Sugeno inference system using Karnik-Mendel algorithm.
    Used for fuzzy sets of type 2.


    :param firings: firings of rules
    :param outputs: outputs of rules
    :param step: size of step used in Karnik-Mendel algorithm
    :return: float that is output of whole inference system
    """
    outputs = outputs.reshape((-1, 1))
    outputs_of_rules = np.concatenate((outputs, firings), axis=1)
    outputs_of_rules = outputs_of_rules[np.argsort(outputs_of_rules[:, 0])]
    domain = np.arange(math.floor(outputs_of_rules[0][0] / step) * step,
                       math.floor(outputs_of_rules[-1][0] / step + 1) * step,
                       step)
    lmf = np.zeros(shape=domain.shape)
    umf = np.zeros(shape=domain.shape)
    for i in range(domain.shape[0]):
        lmf[i] = calculate_membership(domain[i], outputs_of_rules[:, :2])
        umf[i] = calculate_membership(domain[i],
                                      np.concatenate((outputs_of_rules[:, 0].reshape(-1, 1),
                                                      outputs_of_rules[:, 2].reshape(-1, 1)),
                                                     axis=1))

    return karnik_mendel(lmf, umf, domain)


def takagi_sugeno_EIASC(firings: np.ndarray,
                                outputs: np.ndarray) -> float:
    """
        Method of calculating output of Takagi-Sugeno inference system using EIASC algorithm.
        Used for fuzzy sets of type 2.


        :param firings: firings of rules
        :param outputs: outputs of rules
        :return: float that is output of whole inference system
        """
    outputs = outputs.reshape((-1, 1))

    intervals = np.hstack([outputs, outputs, np.array(firings)])
    return EIASC(intervals)


def calculate_membership(x: float,
                         outputs_of_rules: np.ndarray) -> float:
    """
    Calculates values of lower membership function and upper membership function for given element of domain,
    basing on outputs of the rules

    :param x: value from domain for which values are calculated
    :param outputs_of_rules: ndarray with shape nx2, where n is number of records, where first column contains elements
    of domain sorted ascending and second one contains elements from their codomain. All elements are floats.
    :return: returns value of both lower membership function and upper membership function for given x
    """
    if len(outputs_of_rules) == 1:
        if x == outputs_of_rules[0][0]:
            return outputs_of_rules[0][1]
    elif len(outputs_of_rules) > 1:
        if x >= outputs_of_rules[0][0]:
            for i in range(1, len(outputs_of_rules)):
                if x <= outputs_of_rules[i][0]:
                    distance_horizontal = outputs_of_rules[i][0] - outputs_of_rules[i - 1][0]
                    distance_vertical = outputs_of_rules[i][1] - outputs_of_rules[i - 1][1]
                    distance_of_x = x - outputs_of_rules[i - 1][0]
                    horizontal_proportion = distance_of_x / distance_horizontal
                    return distance_vertical * horizontal_proportion + outputs_of_rules[i - 1][1]
    return 0.


def __membership_func_union(mfs: List[np.ndarray]) -> np.ndarray:
    """
    Performs merge of given membership functions by choosing maximum of respective values
    :param mfs: membership functions to unify
    :return: unified membership functions
    """
    if not isinstance(mfs[0], Iterable):
        mfs = [mfs]
    n_functions = len(mfs)
    universe_size = len(mfs[0])
    reshaped_mfs = np.zeros(shape=(n_functions, universe_size))
    for i, mf in enumerate(mfs):
        reshaped_mfs[i] = mf
    union = np.max(reshaped_mfs, axis=0)
    return union
