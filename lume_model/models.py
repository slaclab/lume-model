from abc import ABC, abstractmethod
from typing import Dict

from lume_model.variables import InputVariable, OutputVariable


class SurrogateModel(ABC):
    """
    Base class for the surrogate models that includes abstract predict method, which \\
    must be initialized by children.

    """

    @property
    @abstractmethod
    def input_variables(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def output_variables(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        """
        Abstract prediction method that must be overwritten by inheriting classes.
        """
        pass
