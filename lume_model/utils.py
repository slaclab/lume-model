"""
This module contains utility functions for the saving and subsequent loading of saved
variables.

"""


import pickle
import json
import sys
import yaml
from pydoc import locate
import numpy as np
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


def model_from_yaml(config_file, model_class=None, model_kwargs=None):
    """Creates model from yaml configuration. The model class for initialization may
    either be passed to the function as a kwarg or defined in the config file. This function will
    attempt to import the path specified in the yaml.

    Args:
        config_file: Config file
        model_class: Class for initializing model

    Returns:
        model: Initialized model

    """

    config = yaml.safe_load(config_file)

    # set up the input variables
    input_variables = {}
    if "input_variables" in config:
        for variable in config["input_variables"]:

            variable_config = config["input_variables"][variable]

            # build variable
            if variable_config["type"] == "scalar":
                lume_model_var = ScalarInputVariable(**variable_config)

            elif variable_config["type"] == "image":
                variable_config["default"] = np.load(variable_config["default"])
                variable_config["axis_labels"] = [
                    variable_config["x_label"],
                    variable_config["y_label"],
                ]
                lume_model_var = ImageInputVariable(**variable_config)

            else:
                logger.exception(
                    "Variable type %s not defined.", variable_config["type"],
                )
                sys.exit()

            input_variables[variable] = lume_model_var

    else:
        logger.exception("Input variables are missing from configuration file.")
        sys.exit()

    # set up the output variables

    output_variables = {}
    if "output_variables" in config:
        for variable in config["output_variables"]:

            variable_config = config["output_variables"][variable]

            # build variable
            if variable_config["type"] == "scalar":
                lume_model_var = ScalarOutputVariable(**variable_config)

            elif variable_config["type"] == "image":
                variable_config["default"] = np.load(variable_config["default"])
                variable_config["axis_labels"] = [
                    variable_config["x_label"],
                    variable_config["y_label"],
                ]
                lume_model_var = ImageOutputVariable(**variable_config)

            else:
                logger.exception(
                    "Variable type %s not defined.", variable_config["type"],
                )
                sys.exit()

            output_variables[variable] = lume_model_var

    else:
        logger.exception("Output variables are missing from configuration file.")
        sys.exit()

    if model_class is not None and "model" in config:
        logger.exception(
            "Conflicting class definitions between config file and function argument."
        )
        sys.exit()

    model = None
    model_kwargs = {
        "input_variables": input_variables,
        "output_variables": output_variables,
    }

    if "model" in config:

        # check model requirements before proceeding
        if "requirements" in config["model"]:
            for req in config["model"]["requirements"]:
                module = __import__(req)
                if module:
                    # check for version
                    if isinstance(config["model"]["requirements"][req], (dict,)):
                        version = config["model"]["requirements"][req]
                        if module.version != version:
                            logger.exception(
                                f"Incorrect version for {req}. Model requires {version} and \
                                    {module.version} is installed. Please install the correct \
                                        version to continue."
                            )
                            sys.exit()

                    else:
                        logger.warning(
                            f"No version provided for {req}. Unable to check compatibility."
                        )

                # if requirement not found
                else:
                    logger.warning("Module not installed")

        klass = locate(config["model"]["model_class"])
        if "kwargs" in config["model"]:
            model_kwargs.update(config["model"]["kwargs"])

        if "input_format" in config["model"]:
            model_kwargs["input_format"] = config["model"]["input_format"]

        if "output_format" in config["model"]:
            model_kwargs["output_format"] = config["model"]["output_format"]

        try:
            model = klass(**model_kwargs)
        except:
            logger.exception(f"Unable to load model with args: {model_kwargs}")
            sys.exit()

    elif model_class is not None:
        if model_kwargs:
            model_kwargs.update((model_kwargs))

        try:
            model = model_class(**model_kwargs)
        except:
            logger.exception(f"Unable to load model with args: {model_kwargs}")
            sys.exit()

    return model


def variables_from_yaml(config_file):
    """Returns variables from yaml configuration.

    Args:
        config_file: Yaml file

    Returns:
        tuple

    """

    config = yaml.safe_load(config_file)

    # set up the input variables
    input_variables = {}
    if "input_variables" in config:
        for variable in config["input_variables"]:

            variable_config = config["input_variables"][variable]

            # build variable
            if variable_config["type"] == "scalar":
                lume_model_var = ScalarInputVariable(**variable_config)

            elif variable_config["type"] == "image":
                variable_config["default"] = np.load(variable_config["default"])
                variable_config["axis_labels"] = [
                    variable_config["x_label"],
                    variable_config["y_label"],
                ]
                lume_model_var = ImageInputVariable(**variable_config)

            else:
                logger.exception(
                    "Variable type %s not defined.", variable_config["type"],
                )
                sys.exit()

            input_variables[variable] = lume_model_var

    else:
        logger.exception("Input variables are missing from configuration file.")
        sys.exit()

    # set up the output variables

    output_variables = {}
    if "output_variables" in config:
        for variable in config["output_variables"]:

            variable_config = config["output_variables"][variable]

            # build variable
            if variable_config["type"] == "scalar":
                lume_model_var = ScalarOutputVariable(**variable_config)

            elif variable_config["type"] == "image":
                variable_config["default"] = np.load(variable_config["default"])
                variable_config["axis_labels"] = [
                    variable_config["x_label"],
                    variable_config["y_label"],
                ]
                lume_model_var = ImageOutputVariable(**variable_config)

            else:
                logger.exception(
                    "Variable type %s not defined.", variable_config["type"],
                )
                sys.exit()

            output_variables[variable] = lume_model_var

    return input_variables, output_variables
