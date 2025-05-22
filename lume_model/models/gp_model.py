import os
import logging
import warnings

from pydantic import field_validator

import torch
from torch.distributions import Distribution as TDistribution
from botorch.models import SingleTaskGP, MultiTaskGP, ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal as GPMultivariateNormal
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
    This supports Botorch's SingleTask, MultiTask, and ModelListGP models.

    Note that if the Botorch model was trained with input_transform or outcome_transform set to anything other than None,
    those transformations will be automatically applied, as well as any input_transformers or output_transformers
    passed to this class. For ModelListGP, the passed input_transformers and output_transformers will be applied to all
    models in the list. If different transformers are needed for different models, the models should be instantiated
    separately using Botorch's input_transform and outcome_transform attributes before creating the model list.

    Args:
        model: A single task GPyTorch model or BoTorch model.
        input_transformers: List of input transformers to apply to the input data. Optional, default is None.
        output_transformers: List of output transformers to apply to the output data. Optional, default is None.
    """

    model: SingleTaskGP | MultiTaskGP | ModelListGP
    input_transformers: list[ReversibleInputTransform | torch.nn.Linear] | None = None
    output_transformers: (
        list[OutcomeTransform | ReversibleInputTransform | torch.nn.Linear] | None
    ) = None

    def __init__(self, check_transforms=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if check_transforms:
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
        a warning will be printed to alert the user if transforms are also passed.

        This will only display a warning if the transforms are specified twice. It is up to the user
        to adjust the specified transforms if necessary and reinstantiate.
        """
        # SingleTask and MultiTask
        if (
            hasattr(self.model, "input_transform")
            and self.model.input_transform is not None
        ):
            if self.input_transformers is not None:
                warnings.warn(
                    "Input transforms were passed but the trained model has an input_transform attribute. "
                    "LUME-Model will apply the passed input transforms and use the model's original transform attributes. "
                    "If you'd like to use the passed input transforms only, please retrain your model with input_transform=None. "
                    "To turn off this warning, set check_transforms=False when creating the model."
                )

        if (
            hasattr(self.model, "outcome_transform")
            and self.model.outcome_transform is not None
        ):
            if self.output_transformers is not None:
                warnings.warn(
                    "Output transforms were passed but the trained model has an outcome_transform attribute. "
                    "LUME-Model will apply the passed output transforms and use the model's original transform attributes. "
                    "If you'd like to use the passed output transforms only, please retrain your model with outcome_transform=None. "
                    "To turn off this warning, set check_transforms=False when creating the model."
                )

        # ModelListGP
        if isinstance(self.model, ModelListGP) and (
            self.input_transformers or self.output_transformers
        ):
            # Note that we do not check each model, but instead issue a warning for the entire list
            warnings.warn(
                "The passed input and output transformers will be applied to all models in the ModelListGP. "
                "If any of the models in the list has a transform attribute, it will be used as well on the corresponding model. "
                "To turn off this warning, set check_transforms=False when creating the model."
            )

    def get_input_size(self) -> int:
        """Get the dimensions of the input variables."""
        if isinstance(self.model, SingleTaskGP):
            num_inputs = self.model.train_inputs[0].shape[-1]
        elif isinstance(self.model, MultiTaskGP):
            num_inputs = self.model.train_inputs[0].shape[-1] - 1
        elif isinstance(self.model, ModelListGP):
            if isinstance(self.model.models[0], SingleTaskGP):
                num_inputs = self.model.models[0].train_inputs[0].shape[-1]
            elif isinstance(self.model.models[0], MultiTaskGP):
                num_inputs = self.model.models[0].train_inputs[0].shape[-1] - 1
        else:
            raise ValueError(
                "Model must be an instance of SingleTaskGP or MultiTaskGP."
            )
        return num_inputs

    def get_output_size(self) -> int:
        """Get the dimensions of the output variables."""
        if isinstance(self.model, ModelListGP):
            num_outputs = sum(model.num_outputs for model in self.model.models)
        elif isinstance(self.model, SingleTaskGP) or isinstance(
            self.model, MultiTaskGP
        ):
            num_outputs = self.model.num_outputs
        else:
            raise ValueError(
                "Model must be an instance of SingleTaskGP, MultiTaskGP or ModelListGP."
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
        self.model.eval()
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        return mll(self.model(x), y).item()

    def _get_predictions(
        self,
        input_dict: dict[str, float | torch.Tensor],
        observation_noise: bool = False,
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
        posterior = self._posterior(x, observation_noise=observation_noise)
        # Wrap the distribution in a torch distribution
        distribution = self._get_distribution(posterior)
        # Take mean and covariance of the distribution
        # posterior.mean preserves batch dim, while distribution.mean does not
        mean, covar = posterior.mean, distribution.covariance_matrix
        # Return a dictionary of output variable names to distributions
        return self._create_output_dict((mean, covar))

    def _posterior(self, x: torch.Tensor, observation_noise: bool = False):
        """Compute the posterior distribution.

        Args:
            x: Input tensor.

        Returns:
            Posterior object from the model.
        """
        self.model.eval()
        posterior = self.model.posterior(x, observation_noise=observation_noise)
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

            if self.output_transformers is not None:
                # TODO: make this more robust?
                # If we have two outputs, but transformer has length 1 (e.g. multitask),
                # we should apply the same transform to both outputs
                _mean = self._transform_mean(_mean, i)
                _cov = self._transform_covar(_cov, i)

            output_distributions[name] = GPMultivariateNormal(_mean, _cov)

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

    def _transform_mean(self, mean: torch.Tensor, i) -> torch.Tensor:
        """(Un-)Transforms the model output mean.

        Args:
            mean: Output mean tensor from the model.

        Returns:
            (Un-)Transformed output mean tensor.
        """
        for transformer in self.output_transformers:
            if isinstance(transformer, ReversibleInputTransform):
                try:
                    scale_fac = transformer.coefficient[i]
                    offset = transformer.offset[i]
                except IndexError:
                    # If the transformer has only one coefficient, use it for all outputs
                    # This is needed in the case of multitask models
                    scale_fac = transformer.coefficient[0]
                    offset = transformer.offset[0]
                mean = offset + scale_fac * mean
            elif isinstance(transformer, OutcomeTransform):
                try:
                    scale_fac = transformer.stdvs.squeeze(0)[i]
                    offset = transformer.means.squeeze(0)[i]
                except IndexError:
                    # If the transformer has only one coefficient, use it for all outputs
                    scale_fac = transformer.stdvs.squeeze(0)[0]
                    offset = transformer.means.squeeze(0)[0]
                mean = offset + scale_fac * mean
            else:
                raise NotImplementedError(
                    f"Output transformer {type(transformer)} is not supported."
                )
        return mean

    def _transform_covar(self, cov: torch.Tensor, i: int) -> torch.Tensor:
        """(Un-)Transforms the model output covariance matrix.

        Args:
            cov: Output covariance matrix tensor from the model.
            i: Index of the output variable.

        Returns:
            (Un-)Transformed output covariance matrix tensor.
        """
        for transformer in self.output_transformers:
            if isinstance(transformer, ReversibleInputTransform):
                try:
                    scale_fac = transformer.coefficient[i]
                except IndexError:
                    # If the transformer has only one coefficient, use it for all outputs
                    scale_fac = transformer.coefficient[0]
                scale_fac = scale_fac.expand(cov.shape[:-1])
                scale_mat = DiagLinearOperator(scale_fac)
                cov = scale_mat @ cov @ scale_mat
            elif isinstance(transformer, OutcomeTransform):
                try:
                    scale_fac = transformer.stdvs.squeeze(0)[i]
                except IndexError:
                    # If the transformer has only one coefficient, use it for all outputs
                    scale_fac = transformer.stdvs.squeeze(0)[0]
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
                "Covariance matrix is not positive definite. Attempting to add jitter the diagonal."
            )
            lm = psd_safe_cholesky(cov)  # determines jitter iteratively
            cov = lm @ lm.transpose(-1, -2)

        return cov

    # def dump(
    #     self,
    #     file: Union[str, os.PathLike],
    #     base_key: str = "",
    #     save_models: bool = True,
    #     save_jit: bool = False,
    # ):
    #     """Dump the model to a file.
    #
    #     Note that when dumping ModelListGP, the models in the list are not saved separately.
    #
    #     Args:
    #         file: Path to the file to save the model.
    #         base_key: Base key for the model.
    #         save_models: Whether to save the models.
    #         save_jit: Whether to save the JIT models. Note that this is not supported nor recommended for GP models.
    #     """
    #     # if isinstance(self.model, ModelListGP):
    #     #     # Save each model in the list
    #     #     mod_file = file.split(".yaml")[0].split(".yml")[0]
    #     #     for idx, model in enumerate(self.model.models):
    #     #         model.dump(
    #     #             f"{mod_file}_{idx}.yml",
    #     #             base_key=base_key,
    #     #             save_models=False,  # will be saved in the ensemble
    #     #             save_jit=False,
    #     #         )
    #
    #     # Save the ensemble of models
    #     super().dump(file, base_key, save_models, save_jit)
