"""
This module contains the surrogate model class used for running the online models.

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
