from functools import partial
from abc import ABC, abstractmethod
from typing import Any, Union

import torch
from torch import nn, Tensor
from gpytorch.priors import Prior
from gpytorch.module import Module
from gpytorch.constraints import Interval


class BaseModule(Module, ABC):
    """Abstract base module for calibration."""

    def __init__(
        self,
        model: nn.Module,
        **kwargs,
    ):
        """Initializes BaseModule by registering the model to be calibrated.

        Args:
            model: The model to be calibrated.
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, x):
        pass

    def to(self, device: str):
        self.model.to(device)
        return super().to(device)


class ParameterModule(BaseModule, ABC):
    """Abstract module providing the functionality to register parameters with a prior and constraint."""

    def __init__(
        self,
        model: nn.Module,
        parameter_names: list[str] = None,
        **kwargs,
    ):
        """Initializes ParameterModule by initializing all named parameters.

        Args:
            model: The model to be calibrated.
            parameter_names: A list of parameters to initialize.

        Keyword Args:
            {parameter_name}_size (Union[int, Tuple[int]]): Size of the named parameter. Defaults to 1.
            {parameter_name}_initial (Union[float, Tensor]): Initial value(s) of the named parameter.
              Defaults to zero(s).
            {parameter_name}_default (Union[float, Tensor]): Default value(s) of the named parameter,
              corresponding to zero value(s) of the raw parameter to support regularization during training.
              Defaults to zero(s).
            {parameter_name}_prior (Prior): Prior on named parameter. Defaults to None.
            {parameter_name}_constraint (Interval): Constraint on named parameter. Defaults to None.
            {parameter_name}_mask (Union[Tensor, List]): Boolean mask matching the size of the parameter.
              Allows to select which entries of the unconstrained (raw) parameter are propagated to the constrained
              representation, other entries are set to their default values. This allows to exclude parts of the
              parameter tensor during training. Defaults to None.

        Attributes:
            raw_{parameter_name} (nn.Parameter): Unconstrained parameter tensor.
            {parameter_name} (Union[Tensor]): Parameter tensor transformed according to constraint and
              default value.
            calibration_parameter_names (List[str]): List of named parameters.
            raw_calibration_parameters (List[nn.Parameter]): List of unconstrained parameter tensors.
            calibration_parameters (List[Tensor]): List of parameter tensors transformed according to constraint
              and default value.
        """
        super().__init__(model, **kwargs)
        self.calibration_parameter_names = []
        if parameter_names is None:
            parameter_names = []
        for name in parameter_names:
            self._initialize_parameter(
                name=name,
                size=kwargs.get(f"{name}_size", 1),
                initial=kwargs.get(f"{name}_initial", 0.0),
                default=kwargs.get(f"{name}_default", 0.0),
                prior=kwargs.get(f"{name}_prior"),
                constraint=kwargs.get(f"{name}_constraint"),
                mask=kwargs.get(f"{name}_mask", None),
            )
            self.calibration_parameter_names.append(name)

    def __setstate__(self, state):
        """Defines how to retrieve the class state after unpickling.

        Args:
            state: The unpickled state.
        """
        self.__dict__.update(state)
        for name in self.calibration_parameter_names:
            self._register_parameter_property(name)

    @property
    def raw_calibration_parameters(self) -> list[nn.Parameter]:
        """A list of the raw parameter tensors used for calibration."""
        raw_parameters = []
        for name in self.calibration_parameter_names:
            raw_parameters.append(getattr(self, f"raw_{name}"))
        return raw_parameters

    @property
    def calibration_parameters(self) -> list[Tensor]:
        """A list of the transformed parameter tensors used for calibration."""
        parameters = []
        for name in self.calibration_parameter_names:
            parameters.append(getattr(self, f"{name}"))
        return parameters

    def _initialize_parameter(
        self,
        name: str,
        size: Union[int, tuple[int]],
        initial: Union[float, Tensor],
        default: Union[float, Tensor],
        prior: Prior = None,
        constraint: Interval = None,
        mask: Union[Tensor, list] = None,
    ):
        """Initializes the named parameter.

        Args:
            name: Name of the parameter.
            size: Size of the named parameter.
            initial: Initial value(s) of the named parameter.
            default: Default value(s) of the named parameter.
            prior: Prior on the named parameter.
            constraint: Constraint on the named parameter.
            mask: Boolean mask applied to the transformation from unconstrained (raw) parameter to the
              constrained representation.
        """
        # define initial and default value(s)
        for value, value_str in zip([initial, default], ["initial", "default"]):
            if not isinstance(value, Tensor):
                if isinstance(value, list):
                    value = torch.tensor(value)
                else:
                    value = float(value) * torch.ones(size)
            value_size = value.shape
            if value.dim() == 1 and isinstance(size, int):
                value_size = value.shape[0]
            if not value_size == size:
                raise ValueError(f"Size of {value_str} value tensor is not {size}!")
            setattr(self, f"_{name}_{value_str}", value.clone().detach())
        # create parameter
        if mask is not None and not isinstance(mask, Tensor):
            mask = torch.as_tensor(mask)
        setattr(self, f"{name}_mask", mask)
        self.register_parameter(
            f"raw_{name}", nn.Parameter(getattr(self, f"_{name}_initial"))
        )
        if prior is not None:
            self.register_prior(
                f"{name}_prior",
                prior,
                partial(self._param, name),
                partial(self._closure, name),
            )
        if constraint is not None:
            self.register_constraint(f"raw_{name}", constraint)
        self._closure(name, self, getattr(self, f"_{name}_initial").detach().clone())
        # register parameter property
        self._register_parameter_property(name)

    def _register_parameter_property(self, name):
        """Registers a class property defining the transformed version of the named parameter.

        Args:
            name: Name of the parameter.
        """
        setattr(
            self.__class__,
            name,
            property(
                fget=partial(self._param, name), fset=partial(self._closure, name)
            ),
        )

    @staticmethod
    def _param(name: str, m: Module) -> Union[nn.Parameter, Tensor]:
        """Returns the named parameter transformed according to constraint and default value.

        Args:
            name: Name of the parameter.
            m: Module for which the parameter shall be returned.

        Returns:
            The parameter transformed according to the constraint.
        """
        raw_parameter = getattr(m, f"raw_{name}").clone()
        mask = getattr(m, f"{name}_mask")
        if mask is not None:
            raw_parameter[~mask] = torch.zeros(
                torch.count_nonzero(~mask),
                dtype=raw_parameter.dtype,
                device=raw_parameter.device,
            )
        if hasattr(m, f"raw_{name}_constraint"):
            constraint = getattr(m, f"raw_{name}_constraint")
            default_offset = constraint.inverse_transform(
                getattr(m, f"_{name}_default")
            )
            return constraint.transform(
                raw_parameter + default_offset.to(raw_parameter)
            )
        else:
            default_offset = getattr(m, f"_{name}_default")
            return raw_parameter + default_offset.to(raw_parameter.device)

    @staticmethod
    def _closure(name: str, m: Module, value: Union[float, Tensor]):
        """Sets the named parameter of the module to the given value considering constraint and default value.

        Args:
            name: Name of the parameter.
            m: Module for which the parameter shall be set to the given value.
            value: Value(s) the parameter shall be set to.
        """
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value)
        if hasattr(m, f"raw_{name}_constraint"):
            constraint = getattr(m, f"raw_{name}_constraint")
            default_offset = constraint.inverse_transform(
                getattr(m, f"_{name}_default")
            )
            m.initialize(
                **{f"raw_{name}": constraint.inverse_transform(value) - default_offset}
            )
        else:
            default_offset = getattr(m, f"_{name}_default")
            m.initialize(**{f"raw_{name}": value - default_offset})

    @staticmethod
    def _add_parameter_name_to_kwargs(name: str, kwargs: dict[str, Any]):
        """Adds the given name to the parameter list in kwargs.

        Args:
            name: Name of the parameter to add.
            kwargs: Dictionary of keyword arguments.
        """
        parameter_names = kwargs.get("parameter_names")
        if parameter_names is not None:
            if name in parameter_names:
                raise ValueError(f"Parameter {name} already exists!")
            parameter_names.append(name)
        else:
            parameter_names = [name]
        kwargs["parameter_names"] = parameter_names

    def forward(self, x):
        raise NotImplementedError()
