from typing import Union, Any
from abc import abstractmethod

from pydantic import model_validator
import torch
from torch.distributions import Distribution as TDistribution

from lume_model.variables import DistributionVariable
from lume_model.models.utils import InputDictModel, format_inputs, itemize_dict
from lume_model.base import LUMEBaseModel


class ProbModelBaseModel(LUMEBaseModel):  # TODO: brainstorm a better name
    """Abstract base class for probabilistic models.

    This class provides a common interface for probabilistic models. All subclasses need to
     implement a `_get_predictions` method that accepts a dictionary input, and returns a
     dictionary of output names to _distributions_. The distributions should be instances of
    `torch.distributions.Distribution`.

    Attributes:
        output_variables: List of output variables, which should be of DistributionVariable type.
        device: Device on which the model will be evaluated. Defaults to "cpu".
        precision: Precision of the model, either "double" or "single". Defaults to "double".
    """

    output_variables: list[DistributionVariable]
    device: Union[torch.device, str] = "cpu"
    precision: str = "double"

    @model_validator(mode="before")
    def validate_output_variables(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate output variables as DistributionVariable."""
        return values

    # @model_validator(mode="after")
    # def validate_dimensions(self):
    #     # implement details of how we get sizes in child classes
    #     # here do the validation
    #     num_inputs = self.get_input_size()
    #     num_outputs = self.get_output_size()
    #
    #     if len(self.input_variables) != num_inputs:
    #         raise ValueError(
    #             f"The initialized model requires {num_inputs} input variables."
    #         )
    #     if len(self.output_variables) != num_outputs:
    #         raise ValueError(
    #             f"The initialized model requires {num_outputs} output variables."
    #         )
    #     return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No range validation for probabilistic models currently implemented
        # self.input_validation_config = {x: "none" for x in self.input_names}

    @property
    def dtype(
        self,
    ):
        """Returns the data type for the model."""
        if self.precision == "double":
            return torch.double
        elif self.precision == "single":
            return torch.float
        else:
            raise ValueError(
                f"Unknown precision {self.precision}, "
                f"expected one of ['double', 'single']."
            )

    # @abstractmethod
    # def get_input_size(self) -> int:
    #     """Get the dimensions of the input variables."""
    #     pass
    #
    # @abstractmethod
    # def get_output_size(self) -> int:
    #     """Get the dimensions of the output variables."""
    #     pass

    def _create_output_dict(
        self, output: list[TDistribution] | TDistribution
    ) -> dict[str, TDistribution]:
        """Create a dictionary of output variable names to distributions.
        This can be defined at the subclass level to handle multi-dimensional outputs correctly
        for each model type before returning the dictionary with `_get_predictions`.

        Args:
            output: A Distribution instance (single model) or a list of Distribution instances
                corresponding to the multi-dimensional output.

        Returns:
            Dictionary of output variable names to distributions.
        """
        pass

    @staticmethod
    def _create_tensor_from_dict(
        d: dict[str, Union[float, torch.Tensor]],
    ) -> torch.Tensor:
        """Create a 2D tensor from a dictionary of floats and tensors.

        Args:
            d: Dictionary of floats or tensors.

        Returns:
            A Torch Tensor."""
        tensors = []

        for key, value in d.items():
            if isinstance(value, float):
                tensors.append(torch.tensor([value]))
            elif isinstance(value, torch.Tensor):
                tensors.append(value)
            else:
                raise ValueError(
                    f"Value for key '{key}' must be either a float or a torch tensor."
                )

        if all(isinstance(value, float) for value in d.values()):
            # All values are floats
            return torch.stack(tensors, dim=1)
        elif all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            lengths = [tensor.size(0) for tensor in tensors]
            if len(set(lengths)) != 1:
                raise ValueError("All tensors must have the same length.")
            dim = tensors[0].dim()
            # Stack tensors into a multidimensional tensor
            return torch.stack(tensors, dim=dim)
        else:
            raise ValueError(
                "All values must be either floats or tensors, and all tensors must have the same length."
            )

    @abstractmethod
    def _get_predictions(
        self, input_dict: dict[str, float | torch.Tensor]
    ) -> dict[str, TDistribution]:
        """Get predictions from the model.

        Args:
             input_dict: Dictionary of input variable names to values. Values can be floats or
                `n` or `b x n` (batch mode) torch tensors.
        Returns:
            A dictionary of output variable names to distributions.
        """
        pass

    def _evaluate(
        self, input_dict: dict[str, Union[float, torch.Tensor]]
    ) -> dict[str, TDistribution]:
        """Evaluate the model.

        Args:
            input_dict: Dictionary of input variable names to values. Values can be floats or
                `n` or `b x n` (batch mode) torch tensors.

        Returns:
            A dictionary of output variable names to distributions.
        """
        # Evaluate and get mean and variance for each output
        output_dict = self._get_predictions(input_dict)
        # Split multi-dimensional output into separate distributions and
        # return output dictionary
        return output_dict

    def input_validation(self, input_dict: dict[str, Union[float, torch.Tensor]]):
        """Validates input dictionary before evaluation.

        Args:
            input_dict: Input dictionary to validate.

        Returns:
            Validated input dictionary.
        """
        # validate input type (ints only are cast to floats for scalars)
        validated_input = InputDictModel(input_dict=input_dict).input_dict
        # format inputs as tensors w/o changing the dtype
        formatted_inputs = format_inputs(validated_input)
        # itemize inputs for validation
        itemized_inputs = itemize_dict(formatted_inputs)

        for ele in itemized_inputs:
            # validate values that were in the torch tensor
            # any ints in the torch tensor will be cast to floats by Pydantic
            # but others will be caught, e.g. booleans
            ele = InputDictModel(input_dict=ele).input_dict
            # validate each value based on its var class and config
            super().input_validation(ele)

        # return the validated input dict for consistency w/ casting ints to floats
        if any([isinstance(value, torch.Tensor) for value in validated_input.values()]):
            validated_input = {
                k: v.to(**self._tkwargs).squeeze(-1) for k, v in validated_input.items()
            }

        return validated_input

    def output_validation(self, output_dict: dict[str, TDistribution]):
        """Itemizes tensors before performing output validation."""
        itemized_outputs = itemize_dict(output_dict)
        for ele in itemized_outputs:
            super().output_validation(ele)


class TorchDistributionWrapper(TDistribution):
    """Wraps any distribution to provide a torch.distributions-like interface."""

    def __init__(self, custom_dist):
        """
        Args:
            custom_dist: An instance of a custom distribution with methods:
                          mean, variance, log_prob, sample, and rsample.
        """
        super().__init__()
        self.custom_dist = custom_dist
        self.device = torch.device("cpu")
        self.dtype = torch.double

    @property
    def _tkwargs(self):
        return {"device": self.device, "dtype": self.dtype}

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the custom distribution."""
        attribute_names = ["mean"]
        result, _ = self._get_attr(attribute_names)
        return result

    @property
    def variance(self) -> torch.Tensor:
        """Return the variance of the custom distribution."""
        attribute_names = ["variance", "var", "cov", "covariance", "covariance_matrix"]
        result, attr_name = self._get_attr(attribute_names)

        if attr_name in ["cov", "covariance", "covariance_matrix"]:
            return torch.diagonal(torch.tensor(result))

        return result

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the log probability for a given value."""
        attribute_names = ["log_prob", "log_likelihood", "logpdf"]
        result, _ = self._get_attr(attribute_names, value)
        return result

    # TODO: check fn signature
    def rsample(self, sample_shape: torch.Size()) -> torch.Tensor:
        """Generate reparameterized samples from the custom distribution."""
        # Fallback to sample if rsample is not implemented
        attribute_names = ["rsample", "sample", "rvs"]
        result, _ = self._get_attr(attribute_names, sample_shape)
        return result

    def sample(self, sample_shape: torch.Size()) -> torch.Tensor:
        """Generate samples from the custom distribution (non-differentiable if using sample)."""
        attribute_names = ["sample", "rvs"]
        # Assume non-torch.Distribution takes an integer sample_shape
        sample_shape = sample_shape.numel()
        result, _ = self._get_attr(attribute_names, sample_shape)
        return result

    def __repr__(self):
        return f"TorchDistributionWrapper({self.custom_dist})"

    def _get_attr(self, attribute_names, value=None):
        """Get the first attribute that is found in the distribution."""
        for attr_name in attribute_names:
            attr_value = getattr(self.custom_dist, attr_name, None)
            if attr_value is not None:
                if callable(attr_value):
                    result = attr_value(value) if value is not None else attr_value()
                else:
                    result = attr_value

                return torch.tensor(result, **self._tkwargs), attr_name

        raise AttributeError(
            f"None of the attributes {attribute_names} found in the distribution."
        )
