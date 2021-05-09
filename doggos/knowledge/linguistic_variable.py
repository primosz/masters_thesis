from typing import Sequence

import numpy as np


class Domain:
    """
    Class representing a domain:
    https://en.wikipedia.org/wiki/Domain_of_a_function
    
    Attributes
    --------------------------------------------
    __min_value : float
        a minimum value in the domain
    __max_value : float
        a maximum value in the domain
    __precision : float
        a precision of the domain
            
    Methods
    --------------------------------------------
    domain -> Sequence[float]
        Returns sequence from assigned intervals and precision
    """

    def __init__(self, min_value: float, max_value: float, precision: float):
        """
        Creates a domain.

        :param min_value: minimum value in the domain
        :param max_value: maximum value in the domain
        :param precision: a step
        """
        self.__min_value = min_value
        self.__max_value = max_value
        self.__precision = precision

    @property
    def precision(self) -> float:
        """
        Return domain's precision.

        :return: precision
        """
        return self.__precision

    def __call__(self) -> Sequence[float]:
        """
        Creates a sequence matching given range and precision
        :return: the domain as sequence of floats
        """
        return np.arange(self.min, self.max, self.precision)

    @property
    def min(self) -> float:
        """
        Return a minimum value in the domain.

        :return: minimum
        """
        return self.__min_value

    @property
    def max(self) -> float:
        """
        Returns a maximum value in the domain.

        :return: maximum
        """
        return self.__max_value


class LinguisticVariable:
    """
    Class representing a linguistic variable.
    Linguistic variable is a measurable fragment of reality:
    https://en.wikipedia.org/wiki/Fuzzy_set
    
    Attributes
    --------------------------------------------
    __name : str
        a name of reality fragment
    __domain : Domain
        a domain     
    
    """

    def __init__(self, name: str, domain: Domain):
        """
        Creates linguistic variable with given name and domain

        :param name: name of the linguistic variable
        :param domain: domain 
        """
        if not isinstance(domain, Domain):
            raise TypeError('Linguistic variable requires the domain to be a Domain type')
        self.__name = name
        self.__domain = domain

    @property
    def domain(self) -> Domain:
        """
        Returns the domain

        :return: the domain of the linguistic variable
        """
        return self.__domain

    @property
    def name(self) -> str:
        """
        Returns the name

        :return: the name of the linguistic variable
        """
        return self.__name
