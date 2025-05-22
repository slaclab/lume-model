from typing import Union, Dict

from pydantic import BaseModel, ConfigDict
import torch
from torch.distributions import Distribution


def itemize_dict(
    d: dict[str, Union[float, torch.Tensor, Distribution]],
) -> list[dict[str, Union[float, torch.Tensor]]]:
    """Itemizes the given in-/output dictionary.

    Args:
        d: Dictionary to itemize.

    Returns:
        List of in-/output dictionaries, each containing only a single value per in-/output.
    """
    has_tensors = any([isinstance(value, torch.Tensor) for value in d.values()])
    itemized_dicts = []
    if has_tensors:
        for k, v in d.items():
            for i, ele in enumerate(v.flatten()):
                if i >= len(itemized_dicts):
                    itemized_dicts.append({k: ele.item()})
                else:
                    itemized_dicts[i][k] = ele.item()
    else:
        itemized_dicts = [d]
    return itemized_dicts


def format_inputs(
    input_dict: dict[str, Union[float, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Formats values of the input dictionary as tensors.

    Args:
        input_dict: Dictionary of input variable names to values.

    Returns:
        Dictionary of input variable names to tensors.
    """
    formatted_inputs = {}
    for var_name, value in input_dict.items():
        v = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        formatted_inputs[var_name] = v
    return formatted_inputs


class InputDictModel(BaseModel):
    """Pydantic model for input dictionary validation.

    Attributes:
        input_dict: Input dictionary to validate.
    """

    input_dict: Dict[str, Union[torch.Tensor, float]]

    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True)
