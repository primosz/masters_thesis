import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from doggos.fuzzy_sets import Type1FuzzySet, IntervalType2FuzzySet
from doggos.utils.membership_functions.membership_functions import inv_gaussian_left, gaussian, inv_gaussian_right, \
    get_sigma


class FuzzySetsParams:
    def __init__(self, dataset: pd.DataFrame):
        self.__dataset = dataset
        self.__means = self.initialize_means()

    def generate_3_t1_sets(self, names, center=False, plot=False):
        fuzzy_sets = {}
        for d in self.__dataset:
            if d == "Decision":
                continue
            if center:
                self.__means[d] = 0.5
            mean = self.__means[d]
            sigma = get_sigma(0, mean) if mean <= 0.5 else get_sigma(mean, 1)
            fuzzy_sets[d] = dict(zip(names, [Type1FuzzySet(inv_gaussian_left(self.__means[d], sigma)),
                                             Type1FuzzySet(gaussian(self.__means[d], sigma)),
                                             Type1FuzzySet(inv_gaussian_right(self.__means[d], sigma))]))
            if plot:
                x1 = np.linspace(0, 1, 200)
                mfs = [inv_gaussian_left(self.__means[d], sigma),
                       gaussian(self.__means[d], sigma),
                       inv_gaussian_right(self.__means[d], sigma)]
                small = list(map(lambda x: mfs[0](x), x1))
                medium = list(map(lambda x: mfs[1](x), x1))
                large = list(map(lambda x: mfs[2](x), x1))
                plt.plot(x1, small, 'r', x1, medium, 'b', x1, large, 'g')
                plt.title(f"Feature: {d}, mean: {mean}")
                plt.show()
        return fuzzy_sets

    def generate_3_t2_sets(self, names, sigma_offset, center=False, plot=False):
        fuzzy_sets = {}
        for d in self.__dataset:
            if d == "Decision":
                continue
            if center:
                self.__means[d] = 0.5
            mean = self.__means[d]
            sigma = get_sigma(0, mean) if mean <= 0.5 else get_sigma(mean, 1)
            fuzzy_sets[d] = dict(
                zip(names, [IntervalType2FuzzySet(inv_gaussian_left(self.__means[d], sigma + sigma_offset),
                                                  inv_gaussian_left(self.__means[d], sigma - sigma_offset)),
                            IntervalType2FuzzySet(gaussian(self.__means[d], sigma - sigma_offset),
                                                  gaussian(self.__means[d], sigma + sigma_offset)),
                            IntervalType2FuzzySet(inv_gaussian_right(self.__means[d], sigma + sigma_offset),
                                                  inv_gaussian_right(self.__means[d], sigma - sigma_offset))]))
            if plot:
                x1 = np.linspace(0, 1, 200)
                mfs = [inv_gaussian_left(self.__means[d], sigma + sigma_offset),
                       inv_gaussian_left(self.__means[d], sigma - sigma_offset),
                       gaussian(self.__means[d], sigma - sigma_offset),
                       gaussian(self.__means[d], sigma + sigma_offset),
                       inv_gaussian_right(self.__means[d], sigma + sigma_offset),
                       inv_gaussian_right(self.__means[d], sigma - sigma_offset)]
                small_l = list(map(lambda x: mfs[0](x), x1))
                small_u = list(map(lambda x: mfs[1](x), x1))
                medium_l = list(map(lambda x: mfs[2](x), x1))
                medium_u = list(map(lambda x: mfs[3](x), x1))
                large_l = list(map(lambda x: mfs[4](x), x1))
                large_u = list(map(lambda x: mfs[5](x), x1))
                plt.plot(x1, small_l, 'r', x1, small_u, 'b',
                         x1, medium_l, 'r', x1, medium_u, 'b',
                         x1, large_l, 'r', x1, large_u, 'b')
                plt.title(f"Feature: {d}, mean: {mean}")
                plt.show()
        return fuzzy_sets

    def generate_5_t1_sets(self, names, center=False, plot=False):
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
                sets = [Type1FuzzySet(inv_gaussian_left(mean / 2, sigma)),
                        Type1FuzzySet(gaussian(mean / 2, sigma)),
                        Type1FuzzySet(gaussian(mean, sigma)),
                        Type1FuzzySet(gaussian(mean + mean / 2, sigma)),
                        Type1FuzzySet(inv_gaussian_right(mean + mean / 2, sigma))]
            else:
                sigma = get_sigma((1 + mean) / 2, mean)
                sets = [Type1FuzzySet(inv_gaussian_left(mean - (1 - mean) / 2, sigma)),
                        Type1FuzzySet(gaussian(mean - (1 - mean)/2, sigma)),
                        Type1FuzzySet(gaussian(mean, sigma)),
                        Type1FuzzySet(gaussian((1 + mean)/2, sigma)),
                        Type1FuzzySet(inv_gaussian_right((1 + mean)/2, sigma))
                        ]
            fuzzy_sets[d] = dict(zip(names, sets))
            if plot:
                x1 = np.linspace(0, 1, 200)
                vsmall = list(map(lambda x: sets[0].membership_function(x), x1))
                small = list(map(lambda x: sets[1].membership_function(x), x1))
                medium = list(map(lambda x: sets[2].membership_function(x), x1))
                large = list(map(lambda x: sets[3].membership_function(x), x1))
                vlarge = list(map(lambda x: sets[4].membership_function(x), x1))
                plt.plot(x1, vsmall, 'm', x1, small, 'r', x1, medium, 'b', x1, large, 'g', x1, vlarge, 'black')
                plt.title(f"Feature: {d}, mean: {mean}")
                plt.show()
        return fuzzy_sets

    def generate_5_t2_sets(self, names, sigma_offset, center=False, plot=False):
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
                sets = [IntervalType2FuzzySet(inv_gaussian_left(mean - (1 - mean) / 2, sigma + sigma_offset),
                                              inv_gaussian_left(mean - (1 - mean) / 2, sigma - sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean - (1 - mean) / 2, sigma - sigma_offset),
                                              gaussian(mean - (1 - mean) / 2, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean, sigma - sigma_offset),
                                              gaussian(mean, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian((1 + mean) / 2, sigma - sigma_offset),
                                              gaussian((1 + mean) / 2, sigma + sigma_offset)),
                        IntervalType2FuzzySet(inv_gaussian_right((1 + mean) / 2, sigma + sigma_offset),
                                              inv_gaussian_right((1 + mean) / 2, sigma - sigma_offset))]
            fuzzy_sets[d] = dict(zip(names, sets))
            if plot:
                x1 = np.linspace(0, 1, 200)
                vsmall_l = list(map(lambda x: sets[0].lower_membership_function(x), x1))
                vsmall_u = list(map(lambda x: sets[0].upper_membership_function(x), x1))
                small_l = list(map(lambda x: sets[1].lower_membership_function(x), x1))
                small_u = list(map(lambda x: sets[1].upper_membership_function(x), x1))
                medium_l = list(map(lambda x: sets[2].lower_membership_function(x), x1))
                medium_u = list(map(lambda x: sets[2].upper_membership_function(x), x1))
                large_l = list(map(lambda x: sets[3].lower_membership_function(x), x1))
                large_u = list(map(lambda x: sets[3].upper_membership_function(x), x1))
                vlarge_l = list(map(lambda x: sets[4].lower_membership_function(x), x1))
                vlarge_u = list(map(lambda x: sets[4].upper_membership_function(x), x1))
                plt.plot(x1, vsmall_l, 'r', x1, vsmall_u, 'b',
                         x1, small_l, 'r', x1, small_u, 'b',
                         x1, medium_l, 'r', x1, medium_u, 'b',
                         x1, large_l, 'r', x1, large_u, 'b',
                         x1, vlarge_l, 'r', x1, vlarge_u, 'b')
                plt.title(f"Feature: {d}, mean: {mean}")
                plt.show()
        return fuzzy_sets

    def generate_7_t1_sets(self, names, center=False, plot=False):
        fuzzy_sets = {}
        sets = []
        for d in self.__dataset:
            if d == "Decision":
                continue
            if center:
                self.__means[d] = 0.5
            mean = self.__means[d]
            if mean <= 0.5:
                sigma = get_sigma(0, mean / 3)
                sets = [Type1FuzzySet(inv_gaussian_left(mean/3, sigma)),
                        Type1FuzzySet(gaussian(mean/3, sigma)),
                        Type1FuzzySet(gaussian(2*mean/3, sigma)),
                        Type1FuzzySet(gaussian(mean, sigma)),
                        Type1FuzzySet(gaussian(mean + mean/3, sigma)),
                        Type1FuzzySet(gaussian(5*mean/3, sigma)),
                        Type1FuzzySet(inv_gaussian_right(5*mean/3, sigma))]
            else:
                sigma = get_sigma((1 + 2*mean)/3, mean)
                sets = [Type1FuzzySet(inv_gaussian_left(mean - (1-mean)*2/3, sigma)),
                        Type1FuzzySet(gaussian(mean - (1-mean)*2/3, sigma)),
                        Type1FuzzySet(gaussian(mean - (1-mean)/3, sigma)),
                        Type1FuzzySet(gaussian(mean, sigma)),
                        Type1FuzzySet(gaussian((1 + 2*mean)/3, sigma)),
                        Type1FuzzySet(gaussian(mean + 2/3 * (1-mean), sigma)),
                        Type1FuzzySet(inv_gaussian_right(mean + 2/3 * (1-mean), sigma))
                        ]
            fuzzy_sets[d] = dict(zip(names, sets))
            if plot:
                x1 = np.linspace(0, 1, 200)
                vsmall = list(map(lambda x: sets[0].membership_function(x), x1))
                small = list(map(lambda x: sets[1].membership_function(x), x1))
                smedium = list(map(lambda x: sets[2].membership_function(x), x1))
                medium = list(map(lambda x: sets[3].membership_function(x), x1))
                lmedium = list(map(lambda x: sets[4].membership_function(x), x1))
                large = list(map(lambda x: sets[5].membership_function(x), x1))
                vlarge = list(map(lambda x: sets[6].membership_function(x), x1))
                plt.plot(x1, vsmall, 'm', x1, small, 'r', x1, 'c', smedium,
                         x1, medium, 'b', x1, lmedium, 'w',
                         x1, large, 'g', x1, vlarge, 'black')
                plt.title(f"Feature: {d}, mean: {mean}")
                plt.show()
        return fuzzy_sets

    def generate_7_t2_sets(self, names, sigma_offset, center=False, plot=False):
        fuzzy_sets = {}
        sets = []
        for d in self.__dataset:
            if d == "Decision":
                continue
            if center:
                self.__means[d] = 0.5
            mean = self.__means[d]
            if mean <= 0.5:
                sigma = get_sigma(0, mean / 3)
                sets = [IntervalType2FuzzySet(inv_gaussian_left(mean/3, sigma + sigma_offset),
                                              inv_gaussian_left(mean/3, sigma - sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean/3, sigma - sigma_offset),
                                              gaussian(mean/3, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian(2*mean/3, sigma - sigma_offset),
                                              gaussian(2*mean/3, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean, sigma - sigma_offset),
                                              gaussian(mean, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean + mean/3, sigma - sigma_offset),
                                              gaussian(mean + mean/3, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian(5*mean/3, sigma - sigma_offset),
                                              gaussian(5*mean/3, sigma + sigma_offset)),
                        IntervalType2FuzzySet(inv_gaussian_right(5*mean/3, sigma + sigma_offset),
                                              inv_gaussian_right(5*mean/3, sigma - sigma_offset))]
            else:
                sigma = get_sigma((1 + 2*mean)/3, mean)
                sets = [IntervalType2FuzzySet(inv_gaussian_left(mean - (1-mean)*2/3, sigma + sigma_offset),
                                              inv_gaussian_left(mean - (1-mean)*2/3, sigma)),
                        IntervalType2FuzzySet(gaussian(mean - (1-mean)*2/3, sigma - sigma_offset),
                                              gaussian(mean - (1-mean)*2/3, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean - (1-mean)/3, sigma - sigma_offset),
                                              gaussian(mean - (1-mean)/3, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean, sigma - sigma_offset),
                                              gaussian(mean, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian((1 + 2*mean)/3, sigma - sigma_offset),
                                              gaussian((1 + 2*mean)/3, sigma + sigma_offset)),
                        IntervalType2FuzzySet(gaussian(mean + 2/3 * (1-mean), sigma - sigma_offset),
                                              gaussian(mean + 2/3 * (1-mean), sigma + sigma_offset)),
                        IntervalType2FuzzySet(inv_gaussian_right(mean + 2/3 * (1-mean), sigma + sigma_offset),
                                              inv_gaussian_right(mean + 2/3 * (1-mean), sigma - sigma_offset))]
            fuzzy_sets[d] = dict(zip(names, sets))
            if plot:
                x1 = np.linspace(0, 1, 200)
                vsmall_l = list(map(lambda x: sets[0].lower_membership_function(x), x1))
                vsmall_u = list(map(lambda x: sets[0].upper_membership_function(x), x1))
                small_l = list(map(lambda x: sets[1].lower_membership_function(x), x1))
                small_u = list(map(lambda x: sets[1].upper_membership_function(x), x1))
                smedium_l = list(map(lambda x: sets[2].lower_membership_function(x), x1))
                smedium_u = list(map(lambda x: sets[2].upper_membership_function(x), x1))
                medium_l = list(map(lambda x: sets[3].lower_membership_function(x), x1))
                medium_u = list(map(lambda x: sets[3].upper_membership_function(x), x1))
                lmedium_l = list(map(lambda x: sets[4].lower_membership_function(x), x1))
                lmedium_u = list(map(lambda x: sets[4].upper_membership_function(x), x1))
                large_l = list(map(lambda x: sets[5].lower_membership_function(x), x1))
                large_u = list(map(lambda x: sets[5].upper_membership_function(x), x1))
                vlarge_l = list(map(lambda x: sets[6].lower_membership_function(x), x1))
                vlarge_u = list(map(lambda x: sets[6].upper_membership_function(x), x1))
                plt.plot(x1, vsmall_l, 'r', x1, vsmall_u, 'b',
                         x1, small_l, 'r', x1, small_u, 'b',
                         x1, smedium_l, 'r', x1, smedium_u, 'b',
                         x1, medium_l, 'r', x1, medium_u, 'b',
                         x1, lmedium_l, 'r', x1, lmedium_u, 'b',
                         x1, large_l, 'r', x1, large_u, 'b',
                         x1, vlarge_l, 'r', x1, vlarge_u, 'b')
                plt.title(f"Feature: {d}, mean: {mean}")
                plt.show()
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
