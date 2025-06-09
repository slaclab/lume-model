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


def check_model_type(model: torch.nn.Module) -> str:
    """Checks the type of pytorch.nn.Module and returns a string indicating the type.

    At this moment, it supports CNN 1D, RNN, and raises NotImplementedError for CNN 2D/3D and Transformer layers.
    Other model architectures (e.g. Linear) are marked as "other".
    """
    is_cnn_1d = any(isinstance(layer, (torch.nn.Conv1d)) for layer in model.modules())
    is_cnn_2d_3d = any(
        isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv3d))
        for layer in model.modules()
    )
    is_rnn = any(
        isinstance(layer, (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU))
        for layer in model.modules()
    )
    is_transformer = any(
        isinstance(
            layer,
            (
                torch.nn.Transformer,
                torch.nn.TransformerEncoder,
                torch.nn.TransformerDecoder,
                torch.nn.TransformerEncoderLayer,
                torch.nn.TransformerDecoderLayer,
            ),
        )
        for layer in model.modules()
    )

    if (is_cnn_1d or is_cnn_2d_3d) and is_rnn:
        raise NotImplementedError(
            "Model architecture with both CNN and RNN layers is not supported at this time."
        )
    if is_cnn_2d_3d:
        for layer in model.modules():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv3d)):
                print(layer)
        raise NotImplementedError(
            "Model architecture with 2D/3D CNN layers is not supported at this time."
        )
    if is_transformer:
        raise NotImplementedError(
            "Model architecture with Transformer layers is not supported at this time. "
        )
    if is_cnn_1d:
        return "cnn-1d"
    if is_rnn:
        return "rnn"
    # If none of the above architectures is detected, return "other"
    return "other"
