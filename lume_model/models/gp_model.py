import logging
from typing import Union

import torch
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


    def mean(self, x):
        """Returns the mean of the posterior distribution"""
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(x)
            mean = posterior.mean
        return mean

    def variance(self, x):
        """Returns the variance of the posterior distribution"""
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(x)
            variance = posterior.variance
        return variance

    def sample(self, x):
        """Returns samples from the posterior distribution"""
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(x)
            samples = posterior.rsample()
        return samples

    def mll(self, x, y):
        """ Returns the marginal log-likelihood value"""
        self.model.eval()
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        return mll(self.model(x), y).item()

    def log_likelihood(self, x):
        """Returns the likelihood values for predictions from the likelihood object"""
        self.model.eval()
        self.model.likelihood.eval()
        with torch.no_grad():
            observed_pred = self.model.likelihood(self.model(x))

        return  observed_pred.mean, observed_pred.variance

    def _evaluate(self, x):
        self.model.eval()
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
