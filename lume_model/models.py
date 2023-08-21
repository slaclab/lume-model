import os
import yaml
from typing import Any, Union

registered_models = []

# models requiring torch
try:
    from lume_model.torch import TorchModel, TorchModule
    registered_models += [TorchModel, TorchModule]
except ModuleNotFoundError:
    pass

# models requiring keras
try:
    from lume_model.keras import KerasModel
    registered_models += [KerasModel]
except ModuleNotFoundError:
    pass


def get_model(name: str):
    """Returns the LUME model class for the given name.

    Args:
        name: Name of LUME model class.

    Returns:
        LUME model class for the given name.
    """
    model_lookup = {m.__name__: m for m in registered_models}
    if name not in model_lookup.keys():
        raise KeyError(f"No model named {name}, available models are {list(model_lookup.keys())}")
    return model_lookup[name]


def model_from_dict(config: dict[str, Any]):
    """Creates LUME model from the given configuration dictionary.

    Args:
        config: Model configuration dictionary.

    Returns:
        Created LUME model.
    """
    model_class = get_model(config["model_class"])
    return model_class(config)


def model_from_yaml(yaml_str: Union[str, os.PathLike]):
    """Creates LUME model from the given YAML formatted string or file path.

    Args:
        yaml_str: YAML formatted string or file path.

    Returns:
        Created LUME model.
    """
    if os.path.exists(yaml_str):
        with open(yaml_str) as f:
            yaml_str = f.read()
    config = yaml.safe_load(yaml_str)
    return model_from_dict(config)
