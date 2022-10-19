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
from typing import Tuple, TextIO, Dict
import logging

from lume_model.variables import (
    Variable,
    ScalarInputVariable,
    ScalarOutputVariable,
    ImageInputVariable,
    ImageOutputVariable,
    ArrayInputVariable,
    ArrayOutputVariable,
)
from lume_model.models import BaseModel

logger = logging.getLogger(__name__)


def save_variables(
    input_variables: Dict[str, Variable],
    output_variables: Dict[str, Variable],
    variable_file: str,
) -> None:
    """Save input and output variables to file. Validates that all variable names are
    unique.

    Args:
        input_variables (dict): Dictionary of input variable names to variable.
        output_variables (dict): Dictionary of output variable names to variable.
        variable_file (str): Name of file to save.

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
    """Load variables from the given variable file.

    Args:
        variable_file (str): Name of variable file.

    Returns:
        Tuple[dict]: Tuple of input variable dictionary and output variable dictionary.

    Example:
        ```
        input_variables, output_variables = load_variables("variable_file.pickle")

        ```
    """
    with open(variable_file, "rb") as f:
        variables = pickle.load(f)

        return variables["input_variables"], variables["output_variables"]


def parse_variables(config: dict) -> Tuple[dict]:
    """
    Accepts a yaml config and returns initalized input and output variables.

    Args:
        config (dict): Opened configuration file

    Returns:
        Tuple[dict]: Tuple of input and output variable dicts.

    """
    # set up the input variables
    input_variables = {}
    if "input_variables" in config:
        for variable in config["input_variables"]:

            variable_config = config["input_variables"][variable]
            variable_config["name"] = variable

            # build variable
            if variable_config["type"] == "scalar":
                lume_model_var = ScalarInputVariable(**variable_config)

            elif variable_config["type"] == "array":
                value_type = variable_config.get("value_type")

                if (
                    isinstance(variable_config["default"], (str,))
                    and value_type is not None
                    and value_type == "string"
                ):
                    pass

                if isinstance(variable_config["default"], (str,)) and (
                    value_type is None or value_type != "string"
                ):
                    variable_config["default"] = np.load(variable_config["default"])

                else:
                    variable_config["default"] = np.array(variable_config["default"])

                lume_model_var = ArrayInputVariable(**variable_config)

            elif variable_config["type"] == "image":
                variable_config["default"] = np.load(variable_config["default"])
                variable_config["axis_labels"] = [
                    variable_config["x_label"],
                    variable_config["y_label"],
                ]
                lume_model_var = ImageInputVariable(**variable_config)

            else:
                logger.exception(
                    "Variable type %s not defined.",
                    variable_config["type"],
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
            variable_config["name"] = variable

            # build variable
            if variable_config["type"] == "scalar":
                lume_model_var = ScalarOutputVariable(**variable_config)

            elif variable_config["type"] == "array":
                lume_model_var = ArrayOutputVariable(**variable_config)

            elif variable_config["type"] == "image":
                variable_config["axis_labels"] = [
                    variable_config["x_label"],
                    variable_config["y_label"],
                ]
                lume_model_var = ImageOutputVariable(**variable_config)

            else:
                logger.exception(
                    "Variable type %s not defined.",
                    variable_config["type"],
                )
                sys.exit()

            output_variables[variable] = lume_model_var

    else:
        logger.exception("Output variables are missing from configuration file.")
        sys.exit()

    return input_variables, output_variables


def model_from_yaml(
    config_file: TextIO,
    model_class=None,
    model_kwargs: dict = None,
    load_model: bool = True,
):
    """Creates model from yaml configuration. The model class for initialization may
    either be passed to the function as a kwarg or defined in the config file. This function will
    attempt to import the path specified in the yaml.

    Args:
        config_file (TextIO): Config file
        model_class (BaseModel): Class for initializing model
        model_kwargs (dict): Kwargs for initializing model.
        load_model (bool): If True, will return model. If False, will return model class and model_kwargs.

    Returns:
        model (BaseModel): Initialized model

    """

    config = yaml.safe_load(config_file)
    if not isinstance(config, (dict,)):
        logger.exception("Invalid config file.")
        sys.exit()

    input_variables, output_variables = parse_variables(config)

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

        model_class = locate(config["model"]["model_class"])
        if "kwargs" in config["model"]:

            if "custom_layers" in config["model"]["kwargs"]:
                custom_layers = config["model"]["kwargs"]["custom_layers"]

                # delete key to avoid overwrite
                del config["model"]["kwargs"]["custom_layers"]
                model_kwargs["custom_layers"] = {}

                for layer, import_path in custom_layers.items():
                    layer_class = locate(import_path)

                    if layer_class is not None:
                        model_kwargs["custom_layers"][layer] = layer_class

                    else:
                        logger.exception("Layer class %s not found.", layer)
                        sys.exit()

            model_kwargs.update(config["model"]["kwargs"])

        if "output_format" in config["model"]:
            model_kwargs["output_format"] = config["model"]["output_format"]


    if model_class is None:
        logger.exception("No model class found.")
        sys.exit()

    if load_model:
        try:
            model = model_class(**model_kwargs)
        except:
            logger.exception(f"Unable to load model with args: {model_kwargs}")
            sys.exit()

        return model

    else:
        return model_class, model_kwargs


def variables_from_yaml(config_file: TextIO) -> Tuple[dict]:
    """Returns variables from yaml configuration.

    Args:
        config_file (TextIO): Yaml file

    Returns:
        Tuple[dict]: Tuple of input and output variable dicts.

    """

    config = yaml.safe_load(config_file)
    if not isinstance(config, (dict,)):
        logger.exception("Invalid config file.")
        sys.exit()

    input_variables, output_variables = parse_variables(config)

    return input_variables, output_variables
