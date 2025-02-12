import logging
from typing import Union

import torch
from torch.distributions import Distribution as  TDistribution
from gpytorch.models import ExactGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from lume_model.base import LUMEBaseModel
from lume_model.variables import ScalarVariable

logger = logging.getLogger(__name__)


class GPModel(LUMEBaseModel):
    model: Union[BatchedMultiOutputGPyTorchModel, ExactGP]
    device: Union[torch.device, str] = "cpu"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dtype = torch.double

    # TODO: add properties
    # TODO: add validation

    @property
    def _tkwargs(self):
        return {"device": self.device, "dtype": self.dtype}

    def likelihood(self):
        return self.model.likelihood

    def mll(self, x, y):
        """ Returns the marginal log-likelihood value"""
        self.model.eval()
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        return mll(self.model(x), y).item()

    def _posterior(self, x):
        self.model.eval()
        posterior = self.model.posterior(x)
        return posterior

    def _evaluate(self, x): # make input dict and output dict
        # transform

        # evaluate
        posterior = self._posterior(x)

        # wrap the distribution in a torch distribution
        distribution = self.get_distribution(posterior)

        # untransform and prepare output

        return distribution

    def get_distribution(self, posterior):
        if isinstance(posterior.distribution, TDistribution):
            return posterior.distribution
        else:
            # wrap the distribution in a torch distribution
            return TorchDistributionWrapper(posterior.distribution)


class TorchDistributionWrapper(TDistribution):
    """Wraps any distribution to provide a torch.distributions-like interface."""
    def __init__(self, custom_dist):
        """
        Args:
            custom_dist: An instance of a custom distribution with methods like mean, variance,
                          log_prob, sample, and rsample.
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
