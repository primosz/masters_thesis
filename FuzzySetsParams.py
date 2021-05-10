import pandas as pd

from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet
from doggos.utils.membership_functions.membership_functions import inv_gaussian_left, gaussian, inv_gaussian_right, \
    get_sigma
import matplotlib.pyplot as plt


class FuzzySetsParams:
    def __init__(self, dataset: pd.DataFrame):
        self.__dataset = dataset
        self.__means = self.initialize_means()

    def generate_t1_sets(self, names):
        fuzzy_sets = {}
        sigma = get_sigma(0, .5)
        for d in self.__dataset:
            if d == "Decision":
                continue
            fuzzy_sets[d] = zip(names, [Type1FuzzySet(inv_gaussian_left(self.__means[d], sigma)),
                                        Type1FuzzySet(gaussian(self.__means[d], sigma)),
                                        Type1FuzzySet(inv_gaussian_right(self.__means[d], sigma))])
        return fuzzy_sets

    def generate_t2_mfs_LMF(self, names, sigma_offset):
        fuzzy_sets = {}
        sigma = get_sigma(0, .5)
        for d in self.__dataset:
            if d == "Decision":
                continue
            fuzzy_sets[d] = zip(names, [IntervalType2FuzzySet(inv_gaussian_left(self.__means[d], sigma + sigma_offset),
                                                              inv_gaussian_left(self.__means[d], sigma - sigma_offset)),
                                        IntervalType2FuzzySet(gaussian(self.__means[d], sigma - sigma_offset),
                                                              gaussian(self.__means[d], sigma - sigma_offset)),
                                        IntervalType2FuzzySet(inv_gaussian_right(self.__means[d], sigma + sigma_offset),
                                                              inv_gaussian_right(self.__means[d], sigma - sigma_offset))])
        return fuzzy_sets

    def initialize_means(self):
        means = dict()
        for d in self.__dataset:
            if d == "Decision":
                continue
            means[d] = self.__dataset[d].mean()
        print(means)
        return means
