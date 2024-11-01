import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Union
from types import FunctionType, MethodType
from io import TextIOWrapper

import yaml
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, SerializeAsAny

from lume_model.variables import (
    InputVariable,
    OutputVariable, ScalarInputVariable, ScalarOutputVariable,
)
from lume_model.utils import (
    try_import_module,
    verify_unique_variable_names,
    serialize_variables,
    deserialize_variables,
    variables_from_dict,
    replace_relative_paths,
)

logger = logging.getLogger(__name__)

JSON_ENCODERS = {
    # function/method type distinguished for class members and not recognized as callables
    FunctionType: lambda x: f"{x.__module__}.{x.__qualname__}",
    MethodType: lambda x: f"{x.__module__}.{x.__qualname__}",
    Callable: lambda x: f"{x.__module__}.{x.__qualname__}",
    type: lambda x: f"{x.__module__}.{x.__name__}",
    np.ndarray: lambda x: x.tolist(),
    np.int64: lambda x: int(x),
    np.float64: lambda x: float(x),
}


def process_torch_module(
        module,
        base_key: str = "",
        key: str = "",
        file_prefix: Union[str, os.PathLike] = "",
        save_modules: bool = True,
):
    """Optionally saves the given torch module to file and returns the filename.

    Args:
        base_key: Base key at this stage of serialization.
        key: Key corresponding to the torch module.
        module: The torch module to process.
        file_prefix: Prefix for generated filenames.
        save_modules: Determines whether torch modules are saved to file.

    Returns:
        Filename under which the torch module is (or would be) saved.
    """
    torch = try_import_module("torch")
    filepath_prefix, filename_prefix = os.path.split(file_prefix)
    prefixes = [ele for ele in [filename_prefix, base_key] if not ele == ""]
    filename = "{}.pt".format(key)
    if prefixes:
        filename = "_".join((*prefixes, filename))
    filepath = os.path.join(filepath_prefix, filename)
    if save_modules:
        torch.save(module, filepath)
    return filename

def recursive_serialize(
        v: dict[str, Any],
        base_key: str = "",
        file_prefix: Union[str, os.PathLike] = "",
        save_models: bool = True,
):
    """Recursively performs custom serialization for the given object.

    Args:
        v: Object to serialize.
        base_key: Base key at this stage of serialization.
        file_prefix: Prefix for generated filenames.
        save_models: Determines whether models are saved to file.

    Returns:
        Serialized object.
    """
    # try to import modules for LUMEBaseModel child classes
    torch = try_import_module("torch")
    # serialize
    v = serialize_variables(v)
    for key, value in v.items():
        if isinstance(value, dict):
            v[key] = recursive_serialize(value, key)
        elif torch is not None and isinstance(value, torch.nn.Module):
            v[key] = process_torch_module(value, base_key, key, file_prefix,
                                          save_models)
        elif isinstance(value, list) and torch is not None and any(
                isinstance(ele, torch.nn.Module) for ele in value):
            v[key] = [
                process_torch_module(value[i], base_key, f"{key}_{i}", file_prefix,
                                     save_models)
                for i in range(len(value))
            ]
        else:
            for _type, func in JSON_ENCODERS.items():
                if isinstance(value, _type):
                    v[key] = func(value)
        # check to make sure object has been serialized, if not use a generic serializer
        try:
            json.dumps(v[key])
        except (TypeError, OverflowError):
            v[key] = f"{v[key].__module__}.{v[key].__class__.__qualname__}"

    return v


def recursive_deserialize(v):
    """Recursively performs custom deserialization for the given object.

    Args:
        v: Object to deserialize.

    Returns:
        Deserialized object.
    """
    # deserialize
    v = deserialize_variables(v)
    for key, value in v.items():
        if isinstance(value, dict):
            v[key] = recursive_deserialize(value)
    return v


def json_dumps(
        v,
        *,
        base_key="",
        file_prefix: Union[str, os.PathLike] = "",
        save_models: bool = True,
):
    """Serializes variables before dumping with json.

    Args:
        v: Object to dump.
        base_key: Base key for serialization.
        file_prefix: Prefix for generated filenames.
        save_models: Determines whether models are saved to file.

    Returns:
        JSON formatted string.
    """
    v = recursive_serialize(v.model_dump(), base_key, file_prefix, save_models)
    v = json.dumps(v)
    return v


def json_loads(v):
    """Loads JSON formatted string and recursively deserializes the result.

    Args:
        v: JSON formatted string to load.

    Returns:
        Deserialized object.
    """
    v = json.loads(v)
    v = recursive_deserialize(v)
    return v


def parse_config(
        config: Union[dict, str, TextIOWrapper, os.PathLike],
        model_fields: dict = None,
) -> dict:
    """Parses model configuration and returns keyword arguments for model constructor.

    Args:
        config: Model configuration as dictionary, YAML or JSON formatted string, file or file path.
        model_fields: Fields expected by the model (required for replacing relative paths).

    Returns:
        Configuration as keyword arguments for model constructor.
    """
    config_file = None
    if isinstance(config, dict):
        d = config
    else:
        if isinstance(config, TextIOWrapper):
            yaml_str = config.read()
            config_file = os.path.abspath(config.name)
        elif isinstance(config, (str, os.PathLike)) and os.path.exists(config):
            with open(config) as f:
                yaml_str = f.read()
            config_file = os.path.abspath(config)
        else:
            yaml_str = config
        d = recursive_deserialize(yaml.safe_load(yaml_str))
    if config_file is not None:
        config_dir = os.path.dirname(os.path.realpath(config_file))
        d = replace_relative_paths(d, model_fields, config_dir)
    return model_kwargs_from_dict(d)


def model_kwargs_from_dict(config: dict) -> dict:
    """Processes model configuration and returns the corresponding keyword arguments for model constructor.

    Args:
        config: Model configuration.

    Returns:
        Configuration as keyword arguments for model constructor.
    """
    config = deserialize_variables(config)
    if all(key in config.keys() for key in ["input_variables", "output_variables"]):
        config["input_variables"], config["output_variables"] = variables_from_dict(
            config)
    config.pop("model_class", None)
    return config


class LUMEBaseModel(BaseModel, ABC):
    """Abstract base class for models using lume-model variables.

    Inheriting classes must define the evaluate method and variable names must be unique (respectively).
    Models build using this framework will be compatible with the lume-epics EPICS server and associated tools.

    Attributes:
        input_variables: List defining the input variables and their order.
        output_variables: List defining the output variables and their order.
    """
    input_variables: list[SerializeAsAny[InputVariable]]
    output_variables: list[SerializeAsAny[OutputVariable]]

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("input_variables", mode="before")
    def validate_input_variables(cls, value):
        new_value = []
        if isinstance(value, dict):
            for name, val in value.items():
                if isinstance(val, dict):
                    if val["variable_type"] == "scalar":
                        new_value.append(ScalarInputVariable(name=name, **val))
                elif isinstance(val, InputVariable):
                    new_value.append(val)
                else:
                    raise TypeError(f"type {type(val)} not supported")
        elif isinstance(value, list):
            new_value = value

        return new_value

    @field_validator("output_variables", mode="before")
    def validate_output_variables(cls, value):
        new_value = []
        if isinstance(value, dict):
            for name, val in value.items():
                if isinstance(val, dict):
                    if val["variable_type"] == "scalar":
                        new_value.append(ScalarOutputVariable(name=name, **val))
                elif isinstance(val, OutputVariable):
                    new_value.append(val)
                else:
                    raise TypeError(f"type {type(val)} not supported")
        elif isinstance(value, list):
            new_value = value

        return new_value

    def __init__(self, *args, **kwargs):
        """Initializes LUMEBaseModel.

        Args:
            *args: Accepts a single argument which is the model configuration as dictionary, YAML or JSON
              formatted string or file path.
            **kwargs: See class attributes.
        """
        if len(args) == 1:
            if len(kwargs) > 0:
                raise ValueError("Cannot specify YAML string and keyword arguments for LUMEBaseModel init.")
            super().__init__(**parse_config(args[0], self.model_fields))
        elif len(args) > 1:
            raise ValueError(
                "Arguments to LUMEBaseModel must be either a single YAML string "
                "or keyword arguments passed directly to pydantic."
            )
        else:
            super().__init__(**kwargs)

    @field_validator("input_variables", "output_variables")
    def unique_variable_names(cls, value):
        verify_unique_variable_names(value)
        return value

    @property
    def input_names(self) -> list[str]:
        return [var.name for var in self.input_variables]

    @property
    def output_names(self) -> list[str]:
        return [var.name for var in self.output_variables]

    @abstractmethod
    def evaluate(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        pass

    def to_json(self, **kwargs) -> str:
        return json_dumps(self, **kwargs)

    def dict(self, **kwargs) -> dict[str, Any]:
        config = super().model_dump(**kwargs)
        return {"model_class": self.__class__.__name__} | config

    def json(self, **kwargs) -> str:
        result = self.to_json(**kwargs)
        config = json.loads(result)
        config = {"model_class": self.__class__.__name__} | config
        return json.dumps(config)

    def yaml(
            self,
            base_key: str = "",
            file_prefix: str = "",
            save_models: bool = False,
    ) -> str:
        """Serializes the object and returns a YAML formatted string defining the model.

        Args:
            base_key: Base key for serialization.
            file_prefix: Prefix for generated filenames.
            save_models: Determines whether models are saved to file.

        Returns:
            YAML formatted string defining the model.
        """
        output = json.loads(
            self.to_json(
                base_key=base_key,
                file_prefix=file_prefix,
                save_models=save_models,
            )
        )
        s = yaml.dump({"model_class": self.__class__.__name__} | output,
                      default_flow_style=None, sort_keys=False)
        return s

    def dump(
            self,
            file: Union[str, os.PathLike],
            base_key: str = "",
            save_models: bool = True,
    ):
        """Returns and optionally saves YAML formatted string defining the model.

        Args:
            file: File path to which the YAML formatted string and corresponding files are saved.
            base_key: Base key for serialization.
            save_models: Determines whether models are saved to file.
        """
        file_prefix = os.path.splitext(os.path.abspath(file))[0]
        with open(file, "w") as f:
            f.write(
                self.yaml(
                    base_key=base_key,
                    file_prefix=file_prefix,
                    save_models=save_models,
                )
            )

    @classmethod
    def from_file(cls, filename: str):
        if not os.path.exists(filename):
            raise OSError(f"File {filename} is not found.")
        with open(filename, "r") as file:
            return cls.from_yaml(file)

    @classmethod
    def from_yaml(cls, yaml_obj: [str, TextIOWrapper]):
        return cls.model_validate(parse_config(yaml_obj, cls.model_fields))
