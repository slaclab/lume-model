import pickle
from typing import Tuple
import logging

from lume_model.models import SurrogateModel
from lume_model.variables import Variable

logger = logging.getLogger(__name__)


def save_variables(model_class, variable_file) -> None:
    """
    Save model class variables to file.

    Parameters
    ----------
    model_class: SurrogateModel

    variable_file: str
        Name of file to save

    """
    variables = {
        "input_variables": model_class.input_variables,
        "output_variables": model_class.output_variables,
    }

    with open(variable_file, "wb") as f:
        pickle.dump(variables, f)


def load_variables(variable_file) -> Tuple[dict]:
    """
    Load variables from the given variable file.

    Parameters
    ----------
    variable_file: str
        Name of variable file

    Returns
    -------
    tuple
        input variable dict, output variable dict

    """
    with open(variable_file, "rb") as f:
        variables = pickle.load(f)

        return variables["input_variables"], variables["output_variables"]
