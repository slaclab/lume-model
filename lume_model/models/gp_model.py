import logging
from typing import Union

import numpy as np
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO: add properties
    # TODO: add validation

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

    @property
    def mean(self):
        """Return the mean of the custom distribution."""
        if hasattr(self.custom_dist, 'mean'):
            return self.custom_dist.mean()
        raise NotImplementedError("Mean method is not implemented for this distribution.")

    @property
    def variance(self):
        """Return the variance of the custom distribution."""
        if hasattr(self.custom_dist, 'variance'):
            return self.custom_dist.variance()
        elif hasattr(self.custom_dist, 'var'):
            return self.custom_dist.var(value)
        elif hasattr(self.custom_dist, 'cov'):
            # Scipy distributions have cov method
            return np.diagonal(self.custom_dist.cov())
        elif hasattr(self.custom_dist, 'covariance_matrix'):
            return np.diagonal(self.custom_dist.covariance_matrix())
        raise NotImplementedError("Variance method is not implemented for this distribution.")

    def log_prob(self, value):
        """Compute the log probability for a given value."""
        if hasattr(self.custom_dist, 'log_prob'):
            return self.custom_dist.log_prob(value)
        elif hasattr(self.custom_dist, 'log_likelihood'):
            return self.custom_dist.log_likelihood(value)
        elif hasattr(self.custom_dist, 'logpdf'):
            # Scipy distributions have logpdf method
            return self.custom_dist.logpdf(value)
        raise NotImplementedError("Log probability method is not implemented for this distribution.")

    def rsample(self, sample_shape=torch.Size()):
        """Generate reparameterized samples from the custom distribution."""
        if hasattr(self.custom_dist, 'rsample'):
            return self.custom_dist.rsample(sample_shape)
        elif hasattr(self.custom_dist, 'sample'):
            # Fallback to sample if rsample is not implemented
            return self.custom_dist.sample(sample_shape)
        raise NotImplementedError("Sampling method is not implemented for this distribution.")

    def sample(self, sample_shape=torch.Size()):
        """Generate samples from the custom distribution (non-differentiable if using sample)."""
        if hasattr(self.custom_dist, 'sample'):
            return self.custom_dist.sample(sample_shape)
        elif hasattr(self.custom_dist, 'rvs'):
            # Scipy distributions have rvs method
            return self.custom_dist.rvs(size=sample_shape.numel())
        raise NotImplementedError("Sample method is not implemented for this distribution.")

    def __repr__(self):
        return f"TorchDistributionWrapper({self.custom_dist})"
