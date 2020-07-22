"""
This module contains the surrogate model class used for running the online models.
The purpose of this base class is to create a configurable

Requirements for the  model:

The evaluate method must accept a list of input variables (ScalarInputVariable, ImageInputVariable).


Example:
    Hard code variables:
    ```
    class ExampleModel(SurrogateModel):
            input_variables = {
                "input": ScalarInputVariable(name="input", default=1, range=[0.0, 5.0]),\
            }

            output_variables = {
                "output": ScalarOutputVariable(name="output"),
            }

            def evaluate(self, input_variables):

                self.input_variables = {
                    variable.name: variable for variable in input_variables
                }

                self.output_variables["output"].value = (
                    self.input_variables["input"].value * 2
                )

                # return input * 2
                return list(self.output_variables.values())

    ```

    Pass variables to __init__:
    ```
    class ExampleModel(SurrogateModel):

        def __init__(self, input_variables, output_variables):
            self.input_variables = input_variables
            self.output_variables = output_variables

        def evaluate(self, input_variables):

            self.input_variables = {
                variable.name: variable for variable in input_variables
            }

            self.output_variables["output"].value = (
                self.input_variables["input"].value * 2
            )

            # return input * 2
            return list(self.output_variables.values())

    ```

    Load model variables from saved variable file:

    ```
    from lume_model.utils import load_variables

    class ExampleModel(SurrogateModel):

        def __init__(self, variable_file):

            self.input_variables, self.output_variables = load_variables(variable_file)

        def evaluate(self, input_variables):

            self.input_variables = {
                variable.name: variable for variable in input_variables
            }

            self.output_variables["output"].value = (
                self.input_variables["input"].value * 2
            )

            # return input * 2
            return list(self.output_variables.values())

    ```



"""
from abc import ABC, abstractmethod
from typing import Dict
import logging
import numpy as np

from lume_model.variables import InputVariable, OutputVariable
from lume_model.utils import load_variables

logger = logging.getLogger(__name__)


class SurrogateModel(ABC):
    """
    Base class for the surrogate models that includes abstract predict method, which
    must be initialized by children.

    """

    @abstractmethod
    def evaluate(self):
        """
        Abstract evaluate method that must be overwritten by inheriting classes.

        Note:
        Must return lume-model output variables.
        """
        pass
