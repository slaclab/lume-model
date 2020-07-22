"""
This module contains utility functions for model construction, saving, and
the subsequent loading of saved variables.

"""


import pickle
import json
import h5py
from typing import Tuple, List
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


def build_variables_from_description_file(
    description_file, input_extras=None, output_extras=None
):
    """
    Utility function for creating variables from a JSON model description file.

    Args:
        description_file (str): Description filename.

        input_extras (dict): Dictionary of extra attributes to add to input variables.
            For example, default images.

        output_extras (dict): Dictionary of extra attributes to add to output variables.
            For example, default images.

    """
    input_variables = {}
    output_variables = {}

    with open(description_file) as f:
        data = json.load(f)

        # Add extras to the description dicts
        if input_extras:
            for input_item in input_extras:
                data["input_variables"][input_item].update(input_extras[input_item])

        if output_extras:
            for output_item in output_extras:
                data["output_variables"][output_item].update(output_extras[output_item])

        # parse input variables
        for variable in data["input_variables"]:
            variable_data = data["input_variables"][variable]

            if variable_data["variable_type"] == "scalar":
                input_variables[variable] = ScalarInputVariable(
                    name=variable,
                    default=variable_data["default"],
                    units=variable_data["units"],
                    value_range=variable_data["range"],
                    parent=variable_data.get("parent"),
                )

            elif variable_data["variable_type"] == "image":
                input_variables[variable] = ImageInputVariable(
                    name=variable,
                    shape=tuple(variable_data["shape"]),
                    default=variable_data["default"],
                    axis_labels=variable_data["axis_labels"],
                    axis_units=variable_data.get("axis_units"),
                    value_range=variable_data.get("range"),
                    x_min_variable=variable_data.get("x_min_variable"),
                    x_max_variable=variable_data.get("x_max_variable"),
                    y_min_variable=variable_data.get("y_min_variable"),
                    y_max_variable=variable_data.get("y_max_variable"),
                )

            elif not variable_data["variable_type"]:
                logger.exception("No variable type provided for %s", variable)

            else:
                logger.exception(
                    '%s variable type (%s) is not an allowed variable type. Variables may be "image" or "scalar"',
                    variable,
                    variable_data["variable_type"],
                )

        # parse output variables
        for variable in data["output_variables"]:
            variable_data = data["output_variables"][variable]

            if variable_data["variable_type"] == "scalar":
                output_variables[variable] = ScalarOutputVariable(
                    name=variable,
                    default=variable_data["default"],
                    units=variable_data["units"],
                    parent=variable_data.get("parent"),
                )

            elif variable_data["variable_type"] == "image":
                output_variables[variable] = ImageOutputVariable(
                    name=variable,
                    shape=tuple(variable_data["shape"]),
                    default=variable_data.get("default"),
                    axis_labels=variable_data["axis_labels"],
                    axis_units=variable_data.get("axis_units"),
                    value_range=variable_data.get("range"),
                    x_min_variable=variable_data.get("x_min_variable"),
                    x_max_variable=variable_data.get("x_max_variable"),
                    y_min_variable=variable_data.get("y_min_variable"),
                    y_max_variable=variable_data.get("y_max_variable"),
                )

    return input_variables, output_variables


def save_model(
    weight_file: str,  # model
    description_filename: str,
    input_scales: List[float],
    input_offsets: List[float],
    output_scales: List[float],
    output_offsets: List[float],
    variable_filename: str,
    input_variable_extras: dict = None,
    output_variable_extras: dict = None,
    lower: float = -1,
    upper: float = 1,
    arch_json: str = None,
    arch_json_file: bool = True,
):
    """
    Utility function for saving the model.

    Args:
        weight_file (str): Model weight filename.

        description_file (str): Description filename.

        input_scales (List[float]): List of values used for input scaling.

        input_offsets (List[float]): Offsets for input variables.

        output_scales (List[float]): List of values used for output scaling.

        output_offsets (List[float]): Offsets for output variables.

        lower (float): Lower bound on variable scaling.

        upper (float): Upper bound on variable scaling.

        arch_json (str): JSON representation of model architecture.

        arch_json_file (str): File holding JSON representation of model architecture.

        input_variable_extras (dict): Additional keys used for building input variables.

        output_variable_extras (dict): Additional keys used for building output variables.

        variable_filename (str): File for saving variables.

    """

    # Everything will be saved into this h5
    h = h5py.File(filename, "a")

    # Contains model architecture as JSON string
    if JSONFile:
        f = open(archjson, "r")
        json_string = f.read()
        h.attrs.create("JSON", fstr(json_string))
    else:
        h.attrs.create("JSON", fstr(archjson))

    with open(description_file) as d:
        description = dict(json.load(d))

    h.attrs.create("input_ordering", list(description["input_variables"].keys()))
    h.attrs.create("output_ordering", list(description["output_variables"].keys()))
    h.attrs.create("input_scales", input_scales)
    h.attrs.create("input_offsets", input_offsets)
    h.attrs.create("output_scales", output_scales)
    h.attrs.create("output_offsets", output_offsets)
    h.attrs.create("lower", lower)
    h.attrs.create("upper", upper)
    h.close()

    # save variables
    build_variables_from_description_file(
        variable_filename, input_extras, output_extras
    )
