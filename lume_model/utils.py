import os
import sys
import yaml
import importlib
from typing import Union, get_origin, get_args

from lume_model.variables import (
    InputVariable,
    OutputVariable,
    ScalarInputVariable,
    ScalarOutputVariable,
)


def try_import_module(name: str):
    """Tries to import module if required.

    Args:
        name: Module name.

    Returns:
        Imported module if successful, None otherwise.
    """
    if name not in sys.modules:
        try:
            module = importlib.import_module(name)
        except ImportError:
            module = None
    else:
        module = sys.modules[name]
    return module


def verify_unique_variable_names(variables: Union[list[InputVariable], list[OutputVariable]]):
    """Verifies that variable names are unique.

    Raises a ValueError if any reoccurring variable names are found.

    Args:
        variables: List of in- or output variables.
    """
    names = [var.name for var in variables]
    non_unique_names = [name for name in set(names) if names.count(name) > 1]
    if non_unique_names:
        if all(isinstance(var, InputVariable) for var in variables):
            var_str = "Input variable"
        elif all(isinstance(var, OutputVariable) for var in variables):
            var_str = "Output variable"
        else:
            var_str = "Variable"
        raise ValueError(f"{var_str} names {non_unique_names} are not unique.")


def serialize_variables(v: dict):
    """Performs custom serialization for in- and output variables.

    Args:
        v: Object to serialize.

    Returns:
        Dictionary with serialized in- and output variables.
    """
    for key, value in v.items():
        if key in ["input_variables", "output_variables"]:
            if isinstance(value, list):
                v[key] = {var_dict["name"]: {var_k: var_v for var_k, var_v in var_dict.items() if
                                             not (var_k == "name" or var_v is None)} for var_dict in value}
    return v


def deserialize_variables(v):
    """Performs custom deserialization for in- and output variables.

    Args:
        v: Object to deserialize.

    Returns:
        Dictionary with deserialized in- and output variables.
    """
    for key, value in v.items():
        if key in ["input_variables", "output_variables"]:
            if isinstance(value, dict):
                v[key] = [var_dict | {"name": var_name} for var_name, var_dict in value.items()]
    return v


def variables_as_yaml(
        input_variables: list[InputVariable],
        output_variables: list[OutputVariable],
        file: Union[str, os.PathLike] = None,
) -> str:
    """Returns and optionally saves YAML formatted string defining the in- and output variables.

    Args:
        input_variables: List of input variables.
        output_variables: List of output variables.
        file: If not None, YAML formatted string is saved to given file path.

    Returns:
        YAML formatted string defining the in- and output variables.
    """
    for variables in [input_variables, output_variables]:
        verify_unique_variable_names(variables)
    v = {"input_variables": [var.dict() for var in input_variables],
         "output_variables": [var.dict() for var in output_variables]}
    s = yaml.dump(serialize_variables(v), default_flow_style=None, sort_keys=False)
    if file is not None:
        with open(file, "w") as f:
            f.write(s)
    return s


def variables_from_dict(config: dict) -> tuple[list[InputVariable], list[OutputVariable]]:
    """Parses given config and returns in- and output variable lists.

    Args:
        config: Variable configuration.

    Returns:
        In- and output variable lists.
    """
    input_variables, output_variables = [], []
    for key, value in {**config}.items():
        if key in ["input_variables", "output_variables"]:
            for var in value:
                variable_type = var.get("variable_type", var.get("type"))
                if variable_type == "scalar":
                    if key == "input_variables":
                        input_variables.append(ScalarInputVariable(**var))
                    elif key == "output_variables":
                        output_variables.append(ScalarOutputVariable(**var))
                elif variable_type in ["array", "image"]:
                    raise ValueError(f"Parsing of variable type {variable_type} is not yet implemented.")
                else:
                    raise ValueError(f"Unknown variable type {variable_type}.")
    for variables in [input_variables, output_variables]:
        verify_unique_variable_names(variables)
    return input_variables, output_variables


def variables_from_yaml(yaml_obj: Union[str, os.PathLike]) -> tuple[list[InputVariable], list[OutputVariable]]:
    """Parses YAML object and returns in- and output variable lists.

    Args:
        yaml_obj: YAML formatted string or file path.

    Returns:
        In- and output variable lists.
    """
    if os.path.exists(yaml_obj):
        with open(yaml_obj) as f:
            yaml_str = f.read()
    else:
        yaml_str = yaml_obj
    config = deserialize_variables(yaml.safe_load(yaml_str))
    return variables_from_dict(config)


def get_valid_path(
        path: Union[str, os.PathLike],
        directory: Union[str, os.PathLike] = "",
) -> Union[str, os.PathLike]:
    """Validates path exists either as relative or absolute path and returns the first valid option.

    Args:
        path: Path to validate.
        directory: Directory against which relative paths are checked.

    Returns:
        The first valid path option as an absolute path.
    """
    relative_path = os.path.join(directory, path)
    if os.path.exists(relative_path):
        return os.path.abspath(relative_path)
    elif os.path.exists(path):
        return os.path.abspath(path)
    else:
        raise OSError(f"File {path} is not found.")


def replace_relative_paths(
        d: dict,
        model_fields: dict = None,
        directory: Union[str, os.PathLike] = "",
) -> dict:
    """Replaces dictionary entries with absolute paths where the model field annotation is not string or path-like.

    Args:
        d: Dictionary to process.
        model_fields: Model fields dictionary used to check expected type.
        directory: Directory against which relative paths are checked.

    Returns:
        Dictionary with replaced paths.
    """
    if model_fields is None:
        model_fields = {}
    for k, v in d.items():
        if isinstance(v, (str, os.PathLike)):
            if k in model_fields.keys():
                field_types = [model_fields[k].annotation]
                if get_origin(model_fields[k].annotation) is Union:
                    field_types = list(get_args(model_fields[k].annotation))
                if all([t not in field_types for t in [str, os.PathLike]]):
                    d[k] = get_valid_path(v, directory)
        elif isinstance(v, list):
            if k in model_fields.keys():
                field_types = []
                for i, field_type in enumerate(get_args(model_fields[k].annotation)):
                    if get_origin(field_type) is Union:
                        field_types.extend(list(get_args(field_type)))
                    else:
                        field_types.append(field_type)
                for i, ele in enumerate(v):
                    if (isinstance(ele, (str, os.PathLike)) and
                            all([t not in field_types for t in [str, os.PathLike]])):
                        v[i] = get_valid_path(ele, directory)
        elif isinstance(v, dict):
            model_subfields = {
                ".".join(key.split(".")[1:]): value
                for key, value in model_fields.items() if key.startswith(f"{k}.")
            }
            d[k] = replace_relative_paths(v, model_subfields, directory)
    return d
