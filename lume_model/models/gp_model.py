import logging

import torch
from torch.distributions import Distribution as TDistribution
from torch.distributions import MultivariateNormal
from botorch.models import SingleTaskGP, MultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform

from lume_model.models.prob_model_base import (
    ProbModelBaseModel,
    TorchDistributionWrapper,
)


logger = logging.getLogger(__name__)


class GPModel(ProbModelBaseModel):
    """LUME-model class for Single Task GP models, using GPyTorch and BoTorch.

    Args:
        model: A single task GPyTorch model or BoTorch model.
        device: Device on which the model will be evaluated. Defaults to "cpu".
        precision: Precision of the model, either "double" or "single". Defaults to "double".
    """

    model: SingleTaskGP | MultiTaskGP  # TODO: any other types?

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_input_size(self) -> int:
        """Get the dimensions of the input variables."""
        if isinstance(self.model, SingleTaskGP):
            num_inputs = self.model.train_inputs[0].shape[-1]
        elif isinstance(self.model, MultiTaskGP):
            num_inputs = self.model.train_inputs[0].shape[-1] - 1
        else:
            raise ValueError(
                "Model must be an instance of SingleTaskGP or MultiTaskGP."
            )
        return num_inputs

    def get_output_size(self) -> int:
        """Get the dimensions of the output variables."""
        if isinstance(self.model, SingleTaskGP):
            num_outputs = (
                self.model.train_targets.shape[0]
                if len(self.model.train_targets.shape) > 1
                else 1
            )
        elif isinstance(self.model, MultiTaskGP):
            num_outputs = len(self.model._output_tasks)
        else:
            raise ValueError(
                "Model must be an instance of SingleTaskGP or MultiTaskGP."
            )
        return num_outputs

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
        # TODO: add validation for x and y?
        self.model.eval()
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        return mll(self.model(x), y).item()

    def _get_predictions(
        self, input_dict: dict[str, float | torch.Tensor]
    ) -> dict[str, TDistribution]:
        """Get the predictions of the model.
        This implements the abstract method from ProbModelBaseModel.

        Args:
            input_dict: Dictionary of input variable names to values.

        Returns:
            Dictionary of output variable names to distributions.
        """
        # Create tensor from input_dict
        x = super()._create_tensor_from_dict(input_dict)
        # Get the posterior distribution
        posterior = self._posterior(x)
        # Wrap the distribution in a torch distribution
        distribution = self._get_distribution(posterior)
        # Return a dictionary of output variable names to distributions
        return self._create_output_dict(distribution)

    def _posterior(self, x):
        """Compute the posterior distribution.

        Args:
            x: Input tensor.

        Returns:
            Posterior object from the model.
        """
        self.model.eval()
        posterior = self.model.posterior(x)
        return posterior

    def _get_distribution(self, posterior) -> TDistribution:
        """Get the distribution from the posterior.s

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

    def _create_output_dict(
        self, distribution: TDistribution
    ) -> dict[str, TDistribution]:
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
            ss = mean.shape[1] if len(mean.shape) > 2 else mean.shape[0]  # sample size
            cov = (
                distribution.covariance_matrix
            )  # special case, what if this doesn't exist?
            # TODO: adjust based on whether multioutput dist has cov matrix/var or not

            # TODO: check if we need to implement for dists other than MVN?
            batch = mean.shape[0] if len(mean.shape) > 2 else None
            for i, name in enumerate(self.output_names):
                if batch is None:
                    _mean = mean[:, i]
                    _cov = torch.zeros(ss, ss, **self._tkwargs)
                    _cov[:, :ss] = cov[i * ss : (i + 1) * ss, i * ss : (i + 1) * ss]
                else:
                    _mean = mean[:, :, i]
                    _cov = torch.zeros(batch, ss, ss, **self._tkwargs)
                    _cov[:, :ss, :ss] = cov[
                        :, i * ss : (i + 1) * ss, i * ss : (i + 1) * ss
                    ]
                output_distributions[name] = MultivariateNormal(_mean, _cov)

            return output_distributions
