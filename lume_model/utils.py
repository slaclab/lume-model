import pickle
from typing import Dict, List

from lume_model.models import SurrogateModel
from lume_model.variables import Variable


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
        "input_variables": list(model_class.input_variables.values()),
        "output_variables": list(model_class.output_variables.values()),
    }
    with open(variable_file, "wb") as f:
        pickle.dump(variables, f)


def load_variables(variable_file) -> Dict[str, List[Variable]]:
    """
    Load variables from the given variable file.

    Parameters
    ----------
    variable_file: str
        Name of variable file

    """
    with open(variable_file, "rb") as f:
        variables = pickle.load(f)

        return variables
