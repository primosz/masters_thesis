from typing import NoReturn, Dict

from doggos.knowledge import LinguisticVariable
from doggos.knowledge.consequents.consequent import Consequent


class TakagiSugenoConsequent(Consequent):
    """
        Class used to represent a fuzzy rule consequent in Takagi-Sugeno model:
        http://researchhubs.com/post/engineering/fuzzy-system/takagi-sugeno-fuzzy-model.html

        Attributes
        --------------------------------------------
        __function_parameters : Dict[LinguisticVariable, float]
            supplies parameters to consequent output function, which takes form y = ax1 + bx2 + ...
        __bias : float
        __consequent_output : float
            represents crisp output value of consequent function
        __linguistic_variable : represents attribute of output inference

        Methods
        --------------------------------------------

        output: float
            value representing output from calculating consequent function with provided
             input parameters

        Examples:
        --------------------------------------------
        domain = Domain(0, 10, 0.01)
        ling_var_f1 = LinguisticVariable('F1', domain)
        ling_var_f2 = LinguisticVariable('F2', domain)
        output_lv = LinguisticVariable('output', domain)

        ts2 = TakagiSugenoConsequent({ling_var_f1: 2, ling_var_f2: 10}, 1, output_lv)
        output = ts2.output({ling_var_f1: 1, ling_var_f2: 1})
        """

    def __init__(self, function_parameters: Dict[LinguisticVariable, float], bias: float,
                 linguistic_variable: LinguisticVariable):
        """
        Create Rules Consequent used in Takagi-Sugeno Inference System.
        :param function_parameters: Dict[LinguisticVariable, float] of input LinguisticVariable name and parameter used
         for calculating output of consequent function
         bias: value representing constant in consequent function
         linguistic_variable: represent attribute of output inference
        """
        self.__function_parameters = function_parameters
        self.__linguistic_variable = linguistic_variable
        self.__consequent_output = 0
        self.__bias = bias

    @property
    def function_parameters(self) -> Dict[LinguisticVariable, float]:
        """
        Getter of function parameters
        :return: function_parameters
        """
        return self.__function_parameters

    @function_parameters.setter
    def function_parameters(self, new_function_parameters: Dict[LinguisticVariable, float]) -> NoReturn:
        """
        Sets new list of consequent's function parameters
        :param new_function_parameters: new dictionary of consequent's function parameters
        :return: NoReturn
        """
        if (not isinstance(new_function_parameters, dict) or
                not all(isinstance(x, float) or isinstance(x, int) for x in new_function_parameters.values()) or
                not all(isinstance(x, LinguisticVariable) for x in new_function_parameters.keys())):
            raise ValueError("Takagi-Sugeno consequent parameters must be Dict[LinguisticVariable, float]!")
        self.__function_parameters = new_function_parameters

    @property
    def bias(self) -> float:
        """
        Getter of bias parameter
        :return: bias
        """
        return self.__bias

    @bias.setter
    def bias(self, new_bias: float) -> NoReturn:
        """
        Sets new bias parameter
        :param new_bias: new bias float value
        :return: NoReturn
        """
        if not (isinstance(new_bias, float) or isinstance(new_bias, int)):
            raise ValueError("Bias value needs to be float or int!")
        self.__bias = new_bias

    @property
    def linguistic_variable(self) -> LinguisticVariable:
        """
        Getter of linguistic_variable
        :return: linguistic_variable
        """
        return self.__linguistic_variable

    @linguistic_variable.setter
    def linguistic_variable(self, new_linguistic_variable: LinguisticVariable) -> NoReturn:
        """
        Sets new linguistic_variable
        :param new_linguistic_variable: new linguistic_variable
        :return: NoReturn
        """
        if not (isinstance(new_linguistic_variable, LinguisticVariable)):
            raise ValueError("linguistic_variable needs to be instance of LinguisticVariable object!")
        self.__linguistic_variable = new_linguistic_variable

    def output(self, consequent_input: Dict[LinguisticVariable, float]) -> float:
        """
        Return rule output level by calculating consequent function with inputs as variables.
        :param consequent_input: inputs of the inference system in Dict[LinguisticVariable, float], which reflects
         input feature name and value
                IMPORTANT: Each of input variable which will be considered in inference process, needs to have
                corresponding function parameter provided.
        :return: name of feature and crisp rule output value that needs to be used in aggregation process
        """
        self.__consequent_output = self.__bias
        try:
            for key in self.__function_parameters:
                self.__consequent_output += consequent_input[key] * self.__function_parameters[key]
            return self.__consequent_output
        except KeyError:
            raise KeyError("Function parameters contain value for input {0} which was not provided!".format(key.name))
