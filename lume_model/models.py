from abc import ABC, abstractmethod
from typing import Dict
import logging

from lume_model.variables import InputVariable, OutputVariable

logger = logging.getLogger(__name__)


class SurrogateModel(ABC):
    """
    Base class for the surrogate models that includes abstract predict method, which \\
    must be initialized by children.

    """

    @property
    @abstractmethod
    def input_variables(self):
        logger.exception("Input variables not implemented")
        raise NotImplementedError

    @property
    @abstractmethod
    def output_variables(self):
        logger.exception("Output variables not implemented")
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        """
        Abstract evaluate method that must be overwritten by inheriting classes.

        Notes
        -----
        Must return lume-model output variables.
        """
        pass
