import torch
from torch import nn, Tensor
from gpytorch.priors import NormalPrior, GammaPrior
from gpytorch.constraints import Positive

from lume_model.calibrations.torch_model.base import ParameterModule


class InputOffset(ParameterModule):
    """Adds input offset calibration to the model.

    Inputs are offset by a learnable parameter: y = model(x + x_offset).
    """

    def __init__(
        self,
        model: nn.Module,
        **kwargs,
    ):
        """Initializes InputOffset module.

        Args:
            model: The model to be calibrated.

        Keyword Args:
            x_offset_size (Union[int, Tuple[int]]): Size of the x_offset parameter. Defaults to 1.
            x_offset_initial (Union[float, Tensor]): Initial value(s) of the x_offset parameter.
              Defaults to zero(s).
            x_offset_default (Union[float, Tensor]): Default value(s) of the x_offset parameter.
              Defaults to zero(s).
            x_offset_prior (Prior): Prior on x_offset parameter. Defaults to a Normal distribution.
            x_offset_constraint (Interval): Constraint on x_offset parameter. Defaults to None.
            x_offset_mask (Union[Tensor, List]): Boolean mask for x_offset parameter, allowing to exclude parts of
              the parameter during training. Defaults to None.
        """
        parameter_name = "x_offset"
        kwargs.setdefault(f"{parameter_name}_size", 1)
        kwargs.setdefault(f"{parameter_name}_initial", 0.0)
        kwargs.setdefault(f"{parameter_name}_default", 0.0)
        tensor_size = kwargs[f"{parameter_name}_size"]
        if isinstance(tensor_size, int):
            tensor_size = (1, tensor_size)
        kwargs[f"{parameter_name}_prior"] = kwargs.get(
            f"{parameter_name}_prior",
            NormalPrior(loc=torch.zeros(tensor_size), scale=torch.ones(tensor_size)),
        )
        self._add_parameter_name_to_kwargs(parameter_name, kwargs)
        super().__init__(model, **kwargs)

    def input_offset(self, x: Tensor) -> Tensor:
        """Offsets the given input tensor.

        Args:
            x: Input tensor.

        Returns:
            Input tensor with added offset.
        """
        return x + self.x_offset

    def forward(self, x: Tensor) -> Tensor:
        return self.model(self.input_offset(x))


class InputScale(ParameterModule):
    """Adds input scale calibration to the model.

    Inputs are scaled by a learnable parameter: y = model(x_scale * x).
    """

    def __init__(
        self,
        model: nn.Module,
        **kwargs,
    ):
        """Initializes InputScale module.

        Args:
            model: The model to be calibrated.

        Keyword Args:
            x_scale_size (Union[int, Tuple[int]]): Size of the x_scale parameter. Defaults to 1.
            x_scale_initial (Union[float, Tensor]): Initial value(s) of the x_scale parameter.
              Defaults to one(s).
            x_scale_default (Union[float, Tensor]): Default value(s) of the x_scale parameter.
              Defaults to one(s).
            x_scale_prior (Prior): Prior on x_scale parameter. Defaults to a Gamma distribution
              (concentration=2.0, rate=2.0).
            x_scale_constraint (Interval): Constraint on x_scale parameter. Defaults to Positive().
            x_scale_mask (Union[Tensor, List]): Boolean mask for x_scale parameter, allowing to exclude parts of
              the parameter during training. Defaults to None.
        """
        parameter_name = "x_scale"
        kwargs.setdefault(f"{parameter_name}_size", 1)
        kwargs.setdefault(f"{parameter_name}_initial", 1.0)
        kwargs.setdefault(f"{parameter_name}_default", 1.0)
        tensor_size = kwargs[f"{parameter_name}_size"]
        if isinstance(tensor_size, int):
            tensor_size = (1, tensor_size)
        kwargs[f"{parameter_name}_prior"] = kwargs.get(
            # mean=1.0, std=0.5
            f"{parameter_name}_prior",
            GammaPrior(
                concentration=2.0 * torch.ones(tensor_size),
                rate=2.0 * torch.ones(tensor_size),
            ),
        )
        kwargs[f"{parameter_name}_constraint"] = kwargs.get(
            f"{parameter_name}_constraint", Positive()
        )
        self._add_parameter_name_to_kwargs(parameter_name, kwargs)
        super().__init__(model, **kwargs)

    def input_scale(self, x: Tensor) -> Tensor:
        """Scales the given input tensor.

        Args:
            x: Input tensor.

        Returns:
            Scaled input tensor.
        """
        return self.x_scale * x

    def forward(self, x: Tensor) -> Tensor:
        return self.model(self.input_scale(x))


class DecoupledLinearInput(InputOffset, InputScale):
    """Adds decoupled linear input calibration to the model.

    Inputs are passed through decoupled linear calibration nodes with learnable offset and scaling
    parameters: y = model(x_scale * (x + x_offset)).
    """

    def __init__(
        self,
        model: nn.Module,
        x_size: int = None,
        x_mask: Tensor = None,
        **kwargs,
    ):
        """Initializes DecoupledLinearInput module.

        Args:
            model: The model to be calibrated.
            x_size: Overwrites x_offset_size and x_scale_size.
            x_mask: Overwrites x_offset_mask and x_scale_mask.

        Keyword Args:
            Inherited from InputOffset and InputScale.
        """
        if x_size is not None:
            kwargs["x_offset_size"] = x_size
            kwargs["x_scale_size"] = x_size
        if x_mask is not None:
            kwargs["x_offset_mask"] = x_mask
            kwargs["x_scale_mask"] = x_mask
        super().__init__(model, **kwargs)

    def decoupled_linear_input(self, x: Tensor) -> Tensor:
        """Offsets and scales the input tensor (in that order).

        Args:
            x: Input tensor.

        Returns:
            Scaled input tensor with added offset.
        """
        return self.input_scale(self.input_offset(x))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(self.decoupled_linear_input(x))


class OutputOffset(ParameterModule):
    """Adds output offset calibration to the model.

    Outputs are offset by a learnable parameter: y = model(x) + y_offset.
    """

    def __init__(
        self,
        model: nn.Module,
        **kwargs,
    ):
        """Initializes OutputOffset module.

        Args:
            model: The model to be calibrated.

        Keyword Args:
            y_offset_size (Union[int, Tuple[int]]): Size of the y_offset parameter. Defaults to 1.
            y_offset_initial (Union[float, Tensor]): Initial value(s) of the y_offset parameter.
              Defaults to zero(s).
            y_offset_default (Union[float, Tensor]): Default value(s) of the y_offset parameter.
              Defaults to zero(s).
            y_offset_prior (Prior): Prior on y_offset parameter. Defaults to a Normal distribution.
            y_offset_constraint (Interval): Constraint on y_offset parameter. Defaults to None.
            y_offset_mask (Union[Tensor, List]): Boolean mask for y_offset parameter, allowing to exclude parts of
              the parameter during training. Defaults to None.
        """
        parameter_name = "y_offset"
        kwargs.setdefault(f"{parameter_name}_size", 1)
        kwargs.setdefault(f"{parameter_name}_initial", 0.0)
        kwargs.setdefault(f"{parameter_name}_default", 0.0)
        tensor_size = kwargs[f"{parameter_name}_size"]
        if isinstance(tensor_size, int):
            tensor_size = (1, tensor_size)
        kwargs[f"{parameter_name}_prior"] = kwargs.get(
            f"{parameter_name}_prior",
            NormalPrior(loc=torch.zeros(tensor_size), scale=torch.ones(tensor_size)),
        )
        self._add_parameter_name_to_kwargs(parameter_name, kwargs)
        super().__init__(model, **kwargs)

    def output_offset(self, y: Tensor) -> Tensor:
        """Offsets the given output tensor.

        Args:
            y: Output tensor.

        Returns:
            Output tensor with added offset.
        """
        return y + self.y_offset

    def forward(self, x: Tensor) -> Tensor:
        return self.output_offset(self.model(x))


class OutputScale(ParameterModule):
    """Adds output scale calibration to the model.

    Outputs are scaled by a learnable parameter: y = y_scale * model(x).
    """

    def __init__(
        self,
        model: nn.Module,
        **kwargs,
    ):
        """Initializes OutputScale module.

        Args:
            model: The model to be calibrated.

        Keyword Args:
            y_scale_size (Union[int, Tuple[int]]): Size of the y_scale parameter. Defaults to 1.
            y_scale_initial (Union[float, Tensor]): Initial value(s) of the y_scale parameter.
              Defaults to one(s).
            y_scale_default (Union[float, Tensor]): Default value(s) of the y_scale parameter.
              Defaults to one(s).
            y_scale_prior (Prior): Prior on y_scale parameter. Defaults to a Gamma distribution
              (concentration=2.0, rate=2.0).
            y_scale_constraint (Interval): Constraint on y_scale parameter. Defaults to Positive().
            y_scale_mask (Union[Tensor, List]): Boolean mask for y_scale parameter, allowing to exclude parts of
              the parameter during training. Defaults to None.
        """
        parameter_name = "y_scale"
        kwargs.setdefault(f"{parameter_name}_size", 1)
        kwargs.setdefault(f"{parameter_name}_initial", 1.0)
        kwargs.setdefault(f"{parameter_name}_default", 1.0)
        tensor_size = kwargs[f"{parameter_name}_size"]
        if isinstance(tensor_size, int):
            tensor_size = (1, tensor_size)
        kwargs[f"{parameter_name}_prior"] = kwargs.get(
            # mean=1.0, std=0.5
            f"{parameter_name}_prior",
            GammaPrior(
                concentration=2.0 * torch.ones(tensor_size),
                rate=2.0 * torch.ones(tensor_size),
            ),
        )
        kwargs[f"{parameter_name}_constraint"] = kwargs.get(
            f"{parameter_name}_constraint", Positive()
        )
        self._add_parameter_name_to_kwargs(parameter_name, kwargs)
        super().__init__(model, **kwargs)

    def output_scale(self, y: Tensor) -> Tensor:
        """Scales the given output tensor.

        Args:
            y: Output tensor.

        Returns:
            Scaled output tensor.
        """
        return self.y_scale * y

    def forward(self, x: Tensor) -> Tensor:
        return self.output_scale(self.model(x))


class DecoupledLinearOutput(OutputOffset, OutputScale):
    """Adds decoupled linear output calibration to the model.

    Outputs are passed through decoupled linear calibration nodes with learnable offset and scaling
    parameters: y = y_scale * (model(x) + y_offset).
    """

    def __init__(
        self,
        model: nn.Module,
        y_size: int = None,
        y_mask: Tensor = None,
        **kwargs,
    ):
        """Initializes DecoupledLinearOutput module.

        Args:
            model: The model to be calibrated.
            y_size: Overwrites y_offset_size and y_scale_size.

        Keyword Args:
            Inherited from OutputOffset and OutputScale.
        """
        if y_size is not None:
            kwargs["y_offset_size"] = y_size
            kwargs["y_scale_size"] = y_size
        if y_mask is not None:
            kwargs["y_offset_mask"] = y_mask
            kwargs["y_scale_mask"] = y_mask
        super().__init__(model, **kwargs)

    def decoupled_linear_output(self, y: Tensor) -> Tensor:
        """Offsets and scales the output tensor (in that order).

        Args:
            y: Output tensor.

        Returns:
            Scaled output tensor with added offset.
        """
        return self.output_scale(self.output_offset(y))

    def forward(self, x: Tensor) -> Tensor:
        return self.decoupled_linear_output(self.model(x))


class DecoupledLinear(DecoupledLinearInput, DecoupledLinearOutput):
    """Adds decoupled linear in- and output calibration to the model.

    In- and outputs are passed through decoupled linear calibration nodes with learnable offset and scaling
    parameters: y = y_scale * (model(x_scale * (x + x_offset)) + y_offset).
    """

    def __init__(
        self,
        model: nn.Module,
        **kwargs,
    ):
        """Initializes DecoupledLinear module.

        Args:
            model: The model to be calibrated.

        Keyword Args:
            Inherited from DecoupledLinearInput and DecoupledLinearOutput.
        """
        super().__init__(model, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        _x = self.decoupled_linear_input(x)
        return self.decoupled_linear_output(self.model(_x))
