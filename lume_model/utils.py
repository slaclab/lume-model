"""
This module contains utility functions for model construction, saving, and
the subsequent loading of saved variables.

"""


import pickle
from typing import Tuple
import logging

from lume_model.models import SurrogateModel
from lume_model.variables import Variable

logger = logging.getLogger(__name__)


def save_variables(input_variables, output_variables, variable_file: str) -> None:
    """Save model class variables to file.

    Args:
        model_class (SurrogateModel): Model class

        variable_file (str): Filename for saving

    """
    variables = {
        "input_variables": input_variables,
        "output_variables": output_variables,
    }

    with open(variable_file, "wb") as f:
        pickle.dump(variables, f)


def load_variables(variable_file: str) -> Tuple[dict]:
    """ Load variables from the given variable file.

    Args:
        variable_file (str): Name of variable file.

    Returns:
        Tuple of input variable dictionary and output variable dictionary.

    """
    with open(variable_file, "rb") as f:
        variables = pickle.load(f)

        return variables["input_variables"], variables["output_variables"]
