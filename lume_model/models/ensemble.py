import os
import warnings
from typing import Union
from pathlib import Path

from pydantic import field_validator

import torch
from torch.distributions import Normal
from torch.distributions.distribution import Distribution as TDistribution

from lume_model.models.prob_model_base import ProbModelBaseModel
from lume_model.models.torch_model import TorchModel


class NNEnsemble(ProbModelBaseModel):
    """LUME-model class for neural network ensembles.

    This class allows for the evaluation of multiple NN models as an ensemble.

    Args:
        models: List of one or more LUME-model neural network models (TorchModel instances).
    """

    models: list[TorchModel]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn("This class is still under development.")

    @field_validator("models", mode="before")
    def validate_torch_model_list(cls, v):
        if all(isinstance(m, (str, os.PathLike)) for m in v):
            for i, m in enumerate(v):
                fname = m.split("_model.pt")[0]
                if os.path.exists(m) and os.path.exists(f"{fname}.yml"):
                    # if it's a wrapper around TorchModel, might need a different class or a different way to load
                    v[i] = TorchModel(Path(f"{fname}.yml"))
                else:
                    raise OSError(
                        f"Both files, {m} and {fname}.yml, are required to load the models."
                    )

        if not all(isinstance(m, TorchModel) for m in v):
            raise TypeError("All models must be of type TorchModel.")

        return v

    def _get_predictions(
        self, input_dict: dict[str, float | torch.Tensor]
    ) -> dict[str, TDistribution]:
        """Get the predictions of the ensemble of models.
        This implements the abstract method from ProbModelBaseModel.

        Args:
            input_dict: Dictionary of input variable names to values.

        Returns:
            Dictionary of output variable names to distributions.
        """
        predictions = []
        for model in self.models:
            predictions.append(model.evaluate(input_dict))
        # Return a dictionary of output variable names to distributions
        return self._create_output_dict(predictions)

    def _create_output_dict(self, output_list: list) -> dict[str, TDistribution]:
        """Creates the output dictionary from the ensemble output.

        Args:
            output_list: List of output dictionaries.

        Returns:
            Dictionary of output variable names to distributions.
        """
        # Ensemble output is a list of dicts of output names to values
        # need to map them to a dict of output names to distributions
        ensemble_output_dict = {}

        for key in output_list[0]:  # for each named output
            output_tensor = torch.tensor([d[key].tolist() for d in output_list])
            ensemble_output_dict[key] = Normal(
                output_tensor.mean(axis=0),
                torch.sqrt(output_tensor.var(axis=0)),
            )
        return ensemble_output_dict

    @property
    def _tkwargs(self):
        """Returns the device and dtype for the model."""
        return {"device": self.device, "dtype": self.dtype}

    def dump(
        self,
        file: Union[str, os.PathLike],
        base_key: str = "",
        save_models: bool = True,
        save_jit: bool = False,
    ):
        """Dump the model to a file.

        Args:
            file: Path to the file to save the model.
            base_key: Base key for the model.
            save_models: Whether to save the models.
            save_jit: Whether to save the JIT models.
        """
        # Save each model in the ensemble
        mod_file = file.split(".yaml")[0].split(".yml")[0]
        for idx, model in enumerate(self.models):
            model.dump(
                f"{mod_file}_{idx}.yml",
                base_key=base_key,
                save_models=save_models,
                save_jit=save_jit,
            )

        # Save the ensemble of models
        super().dump(file, base_key, save_models, save_jit)
