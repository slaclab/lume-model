"""
This module contains utility functions for the saving and subsequent loading of saved
variables.

"""


import pickle
import json
import sys
from pydoc import locate
from typing import Tuple, List
import logging

from lume_model.variables import (
    Variable,
    ScalarInputVariable,
    ScalarOutputVariable,
    ImageInputVariable,
    ImageOutputVariable,
)

logger = logging.getLogger(__name__)


def save_variables(input_variables, output_variables, variable_file: str) -> None:
    """Save input and output variables to file. Validates that all variable names are
    unique.

    Args:
        model_class (SurrogateModel): Model class

        variable_file (str): Filename for saving

    Example:
        ```
        input_variables = {
            "input1": ScalarInputVariable(name="input1", default=1, range=[0.0, 5.0]),
            "input2": ScalarInputVariable(name="input2", default=2, range=[0.0, 5.0]),
            }

        output_variables = {
            "output1": ScalarOutputVariable(name="output1"),
            "output2": ScalarOutputVariable(name="output2"),
        }

        save_variables(input_variables, output_variables, "variable_file.pickle")
        ```

    """

    # check unique names for all variables
    variable_names = [var.name for var in input_variables.values()]
    variable_names += [var.name for var in output_variables.values()]
    for var in set(variable_names):
        if variable_names.count(var) > 1:
            logger.exception(
                "Duplicate variable name %s. All variables must have unique names.", var
            )
            raise ValueError

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

    Example:
        ```
        input_variables, output_variables = load_variables("variable_file.pickle")

        ```
    """
    with open(variable_file, "rb") as f:
        variables = pickle.load(f)

        return variables["input_variables"], variables["output_variables"]


def model_from_yaml(config_file, model_class=None):
    """Creates model from yaml configuration.

    """
    config = yaml.safe_load(config_file)

    input_variables = []
    if "input_variables" in config:
        for variable in config["input_variables"]:
            if config["input_variables"][variable]["type"] == "scalar":
                lume_model_var = ScalarInputVariable(
                    config["input_variables"][variable]
                )
                input_variables.append(lume_model_var)

            elif config["input_variables"][variable]["type"] == "image":
                lume_model_var = ImageInputVariable(config["input_variables"][variable])
                input_variables.append(lume_model_var)

            else:
                logger.exception(
                    "Variable type %s not defined.",
                    config["input_variables"][variable]["type"],
                )

    else:
        logger.exception("Input variables are missing from configuration file.")
        sys.exit()

    output_variables = []
    if "output_variables" in config:
        for variable in config["output_variables"]:
            if config["output_variables"][variable]["type"] == "scalar":
                lume_model_var = ScalarInputVariable(
                    config["output_variables"][variable]
                )
                output_variables.append(lume_model_var)

            elif config["output_variables"][variable]["type"] == "image":
                lume_model_var = ImageInputVariable(
                    config["output_variables"][variable]
                )
                output_variables.append(lume_model_var)

            else:
                logger.exception(
                    "Variable type %s not defined.",
                    config["output_variables"][variable]["type"],
                )

    else:
        logger.exception("Output variables are missing from configuration file.")
        sys.exit()

    if model_class is not None and "model" in config:
        logger.exception(
            "Conflicting class definitions between config file and function argument."
        )
        sys.exit()

    if "model" in config:
        if config["model"] in dir():
            klass = locate(config["model"])
            Model = klass(**config["args"])

    elif model_class is not None:
        model = model_class(**config["args"])
