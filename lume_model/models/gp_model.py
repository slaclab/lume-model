import logging
from typing import Union, Any

from pydantic import field_validator, model_validator
import torch
from torch.distributions import Distribution as  TDistribution
from torch.distributions import MultivariateNormal
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform

from lume_model.base import LUMEBaseModel
from lume_model.variables import ScalarVariable, DistributionVariable
from lume_model.models.utils import InputDictModel, format_inputs, itemize_dict

logger = logging.getLogger(__name__)


class GPModel(LUMEBaseModel):
    """LUME-model class for Single Task GP models, using GPyTorch and BoTorch.

    Args:
        model: A single task GPyTorch model or BoTorch model.
        device: Device on which the model will be evaluated. Defaults to "cpu".
        precision: Precision of the model, either "double" or "single". Defaults to "double".
    """
    model: SingleTaskGP
    output_variables: list[DistributionVariable]
    device: Union[torch.device, str] = "cpu"
    precision: str = "double"

    @model_validator(mode='before')
    def validate_dimensions(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate the number of input variables to match the model."""
        model = values["model"]
        input_variables = values["input_variables"]
        output_variables = values["output_variables"]

        if model is None:
            raise ValueError("Model attribute is missing.")

        num_inputs = model.train_inputs[0].shape[-1]
        num_outputs = model.train_targets.shape[0] if len(model.train_targets.shape) > 1 else 1

        if len(input_variables) != num_inputs:
            raise ValueError(f"The initialized GPModel requires {num_inputs} input variables.")
        if len(output_variables) != num_outputs:
            raise ValueError(f"The initialized GPModel requires {num_outputs} output variables.")
        return values

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No range validation for GP models currently implemented
        self.input_validation_config = {x: "none" for x in self.input_names}

    @property
    def dtype(self):
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

    @property
    def _tkwargs(self):
        """Returns the device and dtype for the model."""
        return {"device": self.device, "dtype": self.dtype}

    def input_transform(self) -> InputTransform:
        """Returns the input transform of the model."""
        return self.model.input_transform

    def outcome_transform(self) -> OutcomeTransform:
        """Returns the output transform of the model."""
        return self.model.outcome_transform

    def likelihood(self):
        """Returns the likelihood of the model."""
        return self.model.likelihood

    def mll(self, x, y):
        """Returns the marginal log-likelihood value"""
        self.model.eval()
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        return mll(self.model(x), y).item()

    def _posterior(self, x):
        """Compute the posterior distribution."""
        self.model.eval()
        posterior = self.model.posterior(x)
        return posterior

    def _evaluate(self, input_dict: dict[str, Union[float, torch.Tensor]]) -> dict[str, TDistribution]:
        """Evaluate the model.

        Args:
            input_dict: Dictionary of input variable names to values.

        Returns:
            A dictionary of output variable names to distributions.
        """
        # Create input tensor
        input_tensor = self._create_tensor_from_dict(input_dict)
        # Evaluate
        posterior = self._posterior(input_tensor)
        # Wrap the distribution in a torch distribution
        distribution = self._get_distribution(posterior)
        # Split multi-dimensional output into separate distributions and
        # return output dictionary
        return self._create_output_dict(distribution)

    def _get_distribution(self, posterior) -> TDistribution:
        """Get the distribution from the posterior.

        Args:
            posterior: Posterior object from the model.

        Returns:
            A torch distribution object.
        """
        if isinstance(posterior.distribution, TDistribution):
            return posterior.distribution
        else:
            # Wrap the distribution in a torch distribution
            return TorchDistributionWrapper(posterior.distribution)

    @staticmethod
    def _create_tensor_from_dict(d: dict[str, Union[float, torch.Tensor]],
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
                raise ValueError(f"Value for key '{key}' must be either a float or a torch tensor.")

        if all(isinstance(value, float) for value in d.values()):
            # All values are floats
            return torch.stack(tensors, dim=1)
        elif all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            lengths = [tensor.size(0) for tensor in tensors]
            if len(set(lengths)) != 1:
                raise ValueError("All tensors must have the same length.")
            # Stack tensors into a multidimensional tensor
            return torch.stack(tensors, dim=1)
        else:
            raise ValueError(
                "All values must be either floats or tensors, and all tensors must have the same length.")

    def _create_output_dict(self, distribution: TDistribution) -> dict[str, TDistribution]:
        """Returns outputs as dictionary of output names and their corresponding distributions.

        Args:
            distribution: Distribution corresponding to the multi-dimensional output.

        Returns:
            Dictionary of output variable names to distributions.
        """
        if len(self.output_names) == 1:
            return {self.output_names[0]: distribution}
        else:
            # Note: only for independent outputs (SingleTaskGP)
            output_distributions = {}
            mean = distribution.mean
            ss = mean.shape[0]  # sample size
            cov = distribution.covariance_matrix

            # TODO: check if we need to implement for dists other than MVN?
            for i, name in enumerate(self.output_names):
                _mean = mean[:, i]
                _cov = torch.zeros(ss, ss, **self._tkwargs)
                _cov[:, :ss] = cov[i * ss:(i + 1) * ss, i * ss: (i + 1) * ss]
                output_distributions[name] = MultivariateNormal(_mean, _cov)

            return output_distributions

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
            validated_input = {k: v.to(**self._tkwargs) for k, v in validated_input.items()}

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
        attribute_names = ['variance', 'var', 'cov', 'covariance', 'covariance_matrix']
        result, attr_name = self._get_attr(attribute_names)

        if attr_name in ['cov', 'covariance', 'covariance_matrix']:
            return torch.diagonal(torch.tensor(result))

        return result

    def log_prob(self,  value: torch.Tensor) -> torch.Tensor:
        """Compute the log probability for a given value."""
        attribute_names = ["log_prob", "log_likelihood", "logpdf"]
        result, _ = self._get_attr(attribute_names, value)
        return result

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

        raise AttributeError(f"None of the attributes {attribute_names} found in the distribution.")
