"""
This module contains the surrogate model class used for running models using
lume-model variables. Models build using this framework will be compatible with the
lume-epics EPICS server and associated tools. The base class is intentionally
minimal with the purpose of extensibility and customizability to the user's
preferred format.

Requirements for the  model:

input_variables, output_variables: lume-model input and output variables are required
for use with lume-epics tools. The user can optionally define these as class
attributes or design the subclass so that these are passed during initialization
(see example 2). Names of all variables must be unique in order to be served using
the EPICS tools. A utility function for saving these variables, which also enforces
the uniqueness constraint, is provided (lume_model.utils.save_variables).

evaluate: The evaluate method is called by the serving model. Subclasses must
implement the method, accepting a list of input variables and returning a list of the
model's output variables with value attributes updated based on model execution.


Example:
    Example 1, hard coded variables:
    ```
    class ExampleModel(BaseModel):
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

    Example 2, variables passed during __init__:
    ```
    class ExampleModel(BaseModel):

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

    Example 3, load model variables from variable file (saved with lume_model.utils.save_variables):

    ```
    from lume_model.utils import load_variables

    class ExampleModel(BaseModel):

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

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Base class for the surrogate models that includes abstract predict method, which
    must be initialized by children.

    Attributes:
        input_variables (List[InputVariable]): Must be defined after __init__.

        output_variables (List[OutputVariable]): Must be defined after __init__.

    """

    @abstractmethod
    def evaluate(self):
        """
        Abstract evaluate method that must be overwritten by inheriting classes.

        Note:
        Must return lume-model output variables.
        """
        pass
