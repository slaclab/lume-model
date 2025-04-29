import os
import logging
import warnings

from pydantic import field_validator

import torch
from torch.distributions import Distribution as TDistribution
from torch.distributions import MultivariateNormal
from botorch.models import SingleTaskGP, MultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import ReversibleInputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.operators import DiagLinearOperator

from lume_model.models.prob_model_base import (
    ProbModelBaseModel,
    TorchDistributionWrapper,
)


logger = logging.getLogger(__name__)


class GPModel(ProbModelBaseModel):
    """
    LUME-model class for GP models.
    This supports SingleTask and MultiTask models, using GPyTorch and BoTorch.

    Args:
        model: A single task GPyTorch model or BoTorch model.
        input_transformers: List of input transformers to apply to the input data. Optional, default is None.
        output_transformers: List of output transformers to apply to the output data. Optional, default is None.
    """

    model: SingleTaskGP | MultiTaskGP  # TODO: any other types?
    input_transformers: list[ReversibleInputTransform | torch.nn.Linear] = None
    output_transformers: list[
        OutcomeTransform | ReversibleInputTransform | torch.nn.Linear
    ] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_transforms()

    @field_validator("model", mode="before")
    def validate_gp_model(cls, v):
        if isinstance(v, (str, os.PathLike)):
            if os.path.exists(v):
                v = torch.load(v, weights_only=False)
            else:
                raise OSError(f"File {v} is not found.")
        return v

    @field_validator("input_transformers", "output_transformers", mode="before")
    def validate_transformers(cls, v):
        if not isinstance(v, list):
            raise ValueError("Transformers must be passed as list.")
        loaded_transformers = []
        for t in v:
            if isinstance(t, (str, os.PathLike)):
                if os.path.exists(t):
                    t = torch.load(t, weights_only=False)
                else:
                    raise OSError(f"File {t} is not found.")
            loaded_transformers.append(t)
        v = loaded_transformers
        return v

    def check_transforms(self):
        """
        Check the input and output transforms for the model.

        If the trained model already has transform attributes,
        they must match any passed transforms.

        This will only display a warning if the transforms do not match. It is up to the user
        to adjust the specified transforms if necessary and reinstantiate.
        """
        if (
            hasattr(self.model, "input_transform")
            and self.model.input_transform is not None
        ):
            if self.input_transformers is not None:
                if [self.model.input_transform] != self.input_transformers:
                    warnings.warn(
                        "Input transforms do not match the trained model's input transforms.\n"
                        f"Model attr: {self.model.input_transform}\n"
                        f"Passed attr: {self.input_transformers}\n"
                        "LUME-Model uses the passed input transforms only. "
                        "This may lead to inaccurate predictions."
                    )
                # Either way, the internal transform should be removed
                # to avoid double transformations
                # TODO: should this get saved somewhere though, e.g. as metadata?
                delattr(self.model, "input_transform")

        if self.input_transformers is None:
            self.input_transformers = []

        if (
            hasattr(self.model, "outcome_transform")
            and self.model.outcome_transform is not None
        ):
            if self.output_transformers is not None:
                if [self.model.outcome_transform] != self.output_transformers:
                    warnings.warn(
                        "Output transforms do not match the trained model's output transforms.\n"
                        f"Model attr: {self.model.outcome_transform}\n"
                        f"Passed attr: {self.output_transformers}\n"
                        "LUME-Model uses the passed output transforms only. "
                        "This may lead to inaccurate predictions."
                    )
                # Either way, the internal transform should be removed
                # to avoid double transformations
                # TODO: should this get saved somewhere though, e.g. as metadata?
                # TODO: is this how we want to handle this? Wouldn't results be inaccurate if the GP was trained w/
                # TODO: different outcome_transform?
                delattr(self.model, "outcome_transform")

        if self.output_transformers is None:
            self.output_transformers = []

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
        # Transform the input
        if self.input_transformers is not None:
            x = self._transform_inputs(x)
        # Get the posterior distribution
        posterior = self._posterior(x)
        # Wrap the distribution in a torch distribution
        distribution = self._get_distribution(posterior)
        # Take mean and covariance of the distribution
        mean, covar = distribution.mean, distribution.covariance_matrix
        # Return a dictionary of output variable names to distributions
        # this untransforms the mean and covariance before returning
        return self._create_output_dict((mean, covar))

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
        """Get the distribution from the posterior and checks that the covariance_matrix attribute exists.
        Args:
            posterior: Posterior object from the model.

        Returns:
            A torch distribution object.
        """
        if isinstance(posterior.distribution, TDistribution):
            d = posterior.distribution
        else:
            # Wrap the distribution in a torch distribution
            d = TorchDistributionWrapper(posterior.distribution)

        if not hasattr(d, "covariance_matrix"):
            raise ValueError(
                f"The posterior distribution {type(posterior.distribution)} does not have a covariance matrix attribute."
            )

        return d

    def _create_output_dict(
        self, output: tuple[torch.Tensor, torch.Tensor]
    ) -> dict[str, TDistribution]:
        """Returns outputs as dictionary of output names and their corresponding distributions.
        The returned distributions are constructed as torch multivariate normal distributions.
        At the moment, no other distribution types are supported.

        Args:
            output: Tuple containing mean and covariance of the output.

        Returns:
            Dictionary of output variable names to distributions. Distribution is a torch
            multivariate normal distribution.
        """
        output_distributions = {}
        mean, cov = output
        ss = mean.shape[1] if len(mean.shape) > 2 else mean.shape[0]  # sample size

        batch = mean.shape[0] if len(mean.shape) > 2 else None
        for i, name in enumerate(self.output_names):
            if batch is None:
                _mean = mean[:, i] if len(mean.shape) > 1 else mean
                _cov = torch.zeros(ss, ss, **self._tkwargs)
                _cov[:, :ss] = cov[i * ss : (i + 1) * ss, i * ss : (i + 1) * ss]
            else:
                _mean = mean[:, :, i]
                _cov = torch.zeros(batch, ss, ss, **self._tkwargs)
                _cov[:, :ss, :ss] = cov[:, i * ss : (i + 1) * ss, i * ss : (i + 1) * ss]

            # Check that the covariance matrix is positive definite
            _cov = self._check_covariance_matrix(_cov)

            # Last step is to untransform
            if self.output_transformers is not None:
                _mean = self._transform_mean(_mean)
                _cov = self._transform_covar(_cov)

            output_distributions[name] = MultivariateNormal(_mean, _cov)

        return output_distributions

    def _transform_inputs(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Applies transformations to the inputs.

        Args:
            input_tensor: Ordered input tensor to be passed to the transformers.

        Returns:
            Tensor of transformed inputs to be passed to the model.
        """
        for transformer in self.input_transformers:
            if isinstance(transformer, ReversibleInputTransform):
                input_tensor = transformer.transform(input_tensor)
            else:
                input_tensor = transformer(input_tensor)
        return input_tensor

    def _transform_mean(self, mean: torch.Tensor) -> torch.Tensor:
        """(Un-)Transforms the model output mean.

        Args:
            mean: Output mean tensor from the model.

        Returns:
            (Un-)Transformed output mean tensor.
        """
        for transformer in self.output_transformers:
            if isinstance(transformer, ReversibleInputTransform):
                mean = transformer.untransform(mean)
            elif isinstance(transformer, OutcomeTransform):
                scale_fac = transformer.stdvs.squeeze(0)
                offset = transformer.means.squeeze(0)
                mean = offset + scale_fac * mean
            else:
                raise NotImplementedError(
                    f"Output transformer {type(transformer)} is not supported."
                )
        return mean

    def _transform_covar(self, cov: torch.Tensor) -> torch.Tensor:
        """(Un-)Transforms the model output covariance matrix.

        Args:
            cov: Output covariance matrix tensor from the model.

        Returns:
            (Un-)Transformed output covariance matrix tensor.
        """
        for transformer in self.output_transformers:
            if isinstance(transformer, ReversibleInputTransform):
                scale_fac = transformer.coefficient.expand(cov.shape[:-1])
                scale_mat = DiagLinearOperator(scale_fac)
                cov = scale_mat @ cov @ scale_mat
            elif isinstance(transformer, OutcomeTransform):
                scale_fac = transformer.stdvs.squeeze(0)
                scale_fac = scale_fac.expand(cov.shape[:-1])
                scale_mat = DiagLinearOperator(scale_fac)
                cov = scale_mat @ cov @ scale_mat
            else:
                raise NotImplementedError(
                    f"Output transformer {type(transformer)} is not supported."
                )
        return cov

    def _check_covariance_matrix(self, cov: torch.Tensor) -> torch.Tensor:
        """Checks that the covariance matrix is positive definite, and adds jitter if not."""
        try:
            torch.linalg.cholesky(cov)
        except torch._C._LinAlgError:
            warnings.warn(
                f"Covariance matrix is not positive definite. Attempting to add jitter the diagonal."
            )
            l  = psd_safe_cholesky(cov) # determines jitter iteratively
            cov = l @ l.transpose(-1, -2)

        return cov
