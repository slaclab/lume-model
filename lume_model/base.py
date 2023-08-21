import os
import json
import yaml
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Union
from types import FunctionType, MethodType

import numpy as np
from pydantic import BaseModel, validator

from lume_model.variables import (
    InputVariable,
    OutputVariable,
)
from lume_model.utils import (
    try_import_module,
    verify_unique_variable_names,
    serialize_variables,
    deserialize_variables,
    variables_from_dict,
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
    prefixes = [ele for ele in [file_prefix, base_key] if not ele == ""]
    if not prefixes:
        module_name = "{}.pt".format(key)
    else:
        module_name = "{}.pt".format("_".join((*prefixes, key)))
    if save_modules:
        torch.save(module, module_name)
    return module_name


def process_keras_model(
        model,
        base_key: str = "",
        key: str = "",
        file_prefix: Union[str, os.PathLike] = "",
        save_models: bool = True,
):
    """Optionally saves the given keras model to file and returns the filename.

    Args:
        base_key: Base key at this stage of serialization.
        key: Key corresponding to the torch module.
        model: The keras model to process.
        file_prefix: Prefix for generated filenames.
        save_models: Determines whether keras models are saved to file.

    Returns:
        Filename under which the keras model is (or would be) saved.
    """
    prefixes = [ele for ele in [file_prefix, base_key] if not ele == ""]
    if not prefixes:
        model_name = "{}.keras".format(key)
    else:
        model_name = "{}.keras".format("_".join((*prefixes, key)))
    if save_models:
        model.save(model_name)
    return model_name


def recursive_serialize(
        v,
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
    keras = try_import_module("keras")
    # serialize
    v = serialize_variables(v)
    for key, value in v.items():
        if isinstance(value, dict):
            v[key] = recursive_serialize(value, key)
        elif torch is not None and isinstance(value, torch.nn.Module):
            v[key] = process_torch_module(value, base_key, key, file_prefix, save_models)
        elif isinstance(value, list) and torch is not None and any(
                isinstance(ele, torch.nn.Module) for ele in value):
            v[key] = [
                process_torch_module(value[i], base_key, f"{key}_{i}", file_prefix, save_models)
                for i in range(len(value))
            ]
        elif keras is not None and isinstance(value, keras.Model):
            v[key] = process_keras_model(value, base_key, key, file_prefix, save_models)
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
        default,
        base_key="",
        file_prefix: Union[str, os.PathLike] = "",
        save_models: bool = True,
):
    """Serializes variables before dumping with json.

    Args:
        v: Object to dump.
        default: Default for json.dumps().
        base_key: Base key for serialization.
        file_prefix: Prefix for generated filenames.
        save_models: Determines whether models are saved to file.

    Returns:
        JSON formatted string.
    """
    v = recursive_serialize(v, base_key, file_prefix, save_models)
    v = json.dumps(v, default=default)
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


def parse_config(config: Union[dict, str]) -> dict:
    """Parses model configuration and returns keyword arguments for model constructor.

    Args:
        config: Model configuration as dictionary, YAML or JSON formatted string or file path.

    Returns:
        Configuration as keyword arguments for model constructor.
    """
    if isinstance(config, str):
        if os.path.exists(config):
            with open(config) as f:
                yaml_str = f.read()
        else:
            yaml_str = config
        d = recursive_deserialize(yaml.safe_load(yaml_str))
    else:
        d = config
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
        config["input_variables"], config["output_variables"] = variables_from_dict(config)
    _ = config.pop("model_class", None)
    return config


class LUMEBaseModel(BaseModel, ABC):
    """Abstract base class for models using lume-model variables.

    Inheriting classes must define the evaluate method and variable names must be unique (respectively).
    Models build using this framework will be compatible with the lume-epics EPICS server and associated tools.

    Attributes:
        input_variables: List defining the input variables and their order.
        output_variables: List defining the output variables and their order.
    """
    input_variables: list[InputVariable]
    output_variables: list[OutputVariable]

    class Config:
        extra = "allow"
        json_dumps = json_dumps
        json_loads = json_loads
        validate_assignment = True
        arbitrary_types_allowed = True

    def __init__(
            self,
            config: Union[dict, str] = None,
            **kwargs,
    ):
        """Initializes LUMEBaseModel.

        Args:
            config: Model configuration as dictionary, YAML or JSON formatted string or file path. This overrides
              all other arguments.
            **kwargs: See class attributes.
        """
        if config is not None:
            self.__init__(**parse_config(config))
        else:
            super().__init__(**kwargs)

    @validator("input_variables", "output_variables")
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

    def yaml(
            self,
            file: Union[str, os.PathLike] = None,
            save_models: bool = True,
            base_key: str = "",
    ) -> str:
        """Returns and optionally saves YAML formatted string defining the model.

        Args:
            file: If not None, the YAML formatted string is saved to given file path.
            save_models: Determines whether models are saved to file.
            base_key: Base key for serialization.

        Returns:
            YAML formatted string defining the model.
        """
        file_prefix = ""
        if file is not None:
            file_prefix = os.path.splitext(file)[0]
        config = json.loads(self.json(base_key=base_key, file_prefix=file_prefix, save_models=save_models))
        s = yaml.dump({"model_class": self.__class__.__name__} | config,
                      default_flow_style=None, sort_keys=False)
        if file is not None:
            with open(file, "w") as f:
                f.write(s)
        return s
