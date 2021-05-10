import pandas as pd

from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet
from doggos.utils.membership_functions.membership_functions import inv_gaussian_left, gaussian, inv_gaussian_right, \
    get_sigma
import matplotlib.pyplot as plt


class FuzzySetsParams:
    def __init__(self, dataset: pd.DataFrame):
        self.__dataset = dataset
        self.__means = self.initialize_means()

    def generate_t1_sets(self, names, center=False):
        fuzzy_sets = {}
        for d in self.__dataset:
            if d == "Decision":
                continue
            if center:
                self.__means[d] = 0.5
            mean = self.__means[d]
            sigma = get_sigma(0, mean / 2)
            fuzzy_sets[d] = dict(zip(names, [Type1FuzzySet(inv_gaussian_left(self.__means[d], sigma)),
                                             Type1FuzzySet(gaussian(self.__means[d], sigma)),
                                             Type1FuzzySet(inv_gaussian_right(self.__means[d], sigma))]))
        return fuzzy_sets

    def generate_t2_sets(self, names, sigma_offset, center=False):
        fuzzy_sets = {}
        sigma = get_sigma(0, .5)
        for d in self.__dataset:
            if d == "Decision":
                continue
            if center: self.__means[d] = 0.5
            fuzzy_sets[d] = dict(
                zip(names, [IntervalType2FuzzySet(inv_gaussian_left(self.__means[d], sigma + sigma_offset),
                                                  inv_gaussian_left(self.__means[d], sigma - sigma_offset)),
                            IntervalType2FuzzySet(gaussian(self.__means[d], sigma - sigma_offset),
                                                  gaussian(self.__means[d], sigma - sigma_offset)),
                            IntervalType2FuzzySet(inv_gaussian_right(self.__means[d], sigma + sigma_offset),
                                                  inv_gaussian_right(self.__means[d], sigma - sigma_offset))]))
        return fuzzy_sets

    def generate_5_t1_sets(self, names, center=False):
        fuzzy_sets = {}
        for d in self.__dataset:
            if d == "Decision":
                continue
            if center:
                self.__means[d] = 0.5
            mean = self.__means[d]
            sigma = get_sigma(0, mean / 2)
            fuzzy_sets[d] = dict(zip(names, [Type1FuzzySet(inv_gaussian_left(mean / 2, sigma)),
                                             Type1FuzzySet(gaussian(mean / 2, sigma)),
                                             Type1FuzzySet(gaussian(mean, sigma)),
                                             Type1FuzzySet(gaussian(mean + mean / 2, sigma)),
                                             Type1FuzzySet(inv_gaussian_right(mean + mean / 2, sigma))]))
        return fuzzy_sets

    def generate_5_t2_sets(self, names, sigma_offset, center=False):
        fuzzy_sets = {}
        sets = []
        for d in self.__dataset:
            if d == "Decision":
                continue
            if center:
                self.__means[d] = 0.5
            mean = self.__means[d]
            if mean <= 0.5:
                sigma = get_sigma(0, mean / 2)
                sets = [IntervalType2FuzzySet(inv_gaussian_left(mean / 2, sigma + sigma_offset),
                                              inv_gaussian_left(mean / 2, sigma - sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean / 2, sigma - sigma_offset),
                                              gaussian(mean / 2, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean, sigma - sigma_offset),
                                              gaussian(mean, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean + mean / 2, sigma - sigma_offset),
                                              gaussian(mean + mean / 2, sigma + sigma_offset)),
                        IntervalType2FuzzySet(inv_gaussian_right(mean + mean / 2, sigma + sigma_offset),
                                              inv_gaussian_right(mean + mean / 2, sigma - sigma_offset))]
            else:
                sigma = get_sigma((1 + mean) / 2, mean)
                sets = [IntervalType2FuzzySet(inv_gaussian_left(mean - (1 - mean)/2, sigma + sigma_offset),
                                              inv_gaussian_left(mean - (1 - mean)/2, sigma - sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean - (1 - mean)/2, sigma - sigma_offset),
                                              gaussian(mean - (1 - mean)/2, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean, sigma - sigma_offset),
                                              gaussian(mean, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian((1 + mean)/2, sigma - sigma_offset),
                                              gaussian((1 + mean)/2, sigma + sigma_offset)),
                        IntervalType2FuzzySet(inv_gaussian_right((1 + mean)/2, sigma + sigma_offset),
                                              inv_gaussian_right((1 + mean)/2, sigma - sigma_offset))]
            fuzzy_sets[d] = dict(zip(names, sets))
        return fuzzy_sets

    def initialize_means(self):
        means = dict()
        for d in self.__dataset:
            if d == "Decision":
                continue
            means[d] = self.__dataset[d].mean()
            if means[d] < 0.1: means[d] = 0.5
        print(means)
        return means
