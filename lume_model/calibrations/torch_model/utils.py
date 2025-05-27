import torch
from torch import Tensor
from botorch.models.transforms.input import AffineInputTransform

from lume_model.calibrations.torch_model.base import BaseModule


def extract_input_transformer(module: BaseModule) -> AffineInputTransform:
    """Creates an input transformer based on the given calibration module.

    Forward calls of the calibration module correspond to forward calls of the transformer: transformer(x).

    Args:
        module: The calibration module.

    Returns:
          The input transformer.
    """
    x_offset = torch.zeros(1)
    if hasattr(module, "x_offset"):
        x_offset = module.x_offset.detach().clone()
    x_scale = torch.ones(1)
    if hasattr(module, "x_scale"):
        x_scale = module.x_scale.detach().clone()
    return AffineInputTransform(
        d=len(x_offset), coefficient=1 / x_scale, offset=-x_offset
    )


def extract_output_transformer(module: BaseModule) -> AffineInputTransform:
    """Creates an output transformer corresponding to the given calibration module.

    Forward calls of the calibration module correspond to backward calls of the transformer:
    transformer.untransform(x).

    Args:
        module: The calibration module.

    Returns:
        The output transformer.
    """
    y_offset = torch.zeros(1)
    if hasattr(module, "y_offset"):
        y_offset = module.y_offset.detach().clone()
    y_scale = torch.ones(1)
    if hasattr(module, "y_scale"):
        y_scale = module.y_scale.detach().clone()
    return AffineInputTransform(
        d=len(y_offset), coefficient=y_scale, offset=y_scale * y_offset
    )


def extract_transformers(
    module: BaseModule,
) -> (AffineInputTransform, AffineInputTransform):
    """Creates in- and output transformers corresponding to the given calibration module.

    Args:
        module: The calibration module.

    Returns:
        The in- and output transformer.
    """
    return extract_input_transformer(module), extract_output_transformer(module)


def get_decoupled_linear_parameters(
    input_transformer: AffineInputTransform = None,
    output_transformer: AffineInputTransform = None,
) -> dict[str, Tensor]:
    """Returns a parameter dictionary corresponding to the given transformers.

    The created parameter dictionary can be passed to the decoupled linear calibration modules to reproduce
    the following sequence in a forward call:

    x = input_transformer(x)
    x = model(x)
    x = output_transformer.untransform(x)

    Args:
        input_transformer: The input transformer.
        output_transformer: The output transformer.

    Returns:
        A dictionary defining the x_offset, x_scale, y_offset and y_scale parameters.
    """
    parameters = {
        "x_offset": torch.zeros(1),
        "x_scale": torch.ones(1),
        "y_offset": torch.zeros(1),
        "y_scale": torch.ones(1),
    }
    if input_transformer is not None:
        parameters["x_scale"] = 1 / input_transformer.coefficient.detach().clone()
        parameters["x_offset"] = -input_transformer.offset.detach().clone()
    if output_transformer is not None:
        parameters["y_scale"] = output_transformer.coefficient.detach().clone()
        parameters["y_offset"] = (
            output_transformer.offset.detach().clone()
            / output_transformer.coefficient.detach().clone()
        )
    # define sizes and set as initial values
    for key in list(parameters.keys()):
        parameters[f"{key}_size"] = parameters[key].shape
        parameters[f"{key}_initial"] = parameters.pop(key)
    return parameters
