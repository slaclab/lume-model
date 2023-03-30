from typing import Dict, List, Union

import torch
from gpytorch.means import Mean

from lume_model.pytorch import PyTorchModel


class CustomMean(Mean):
    def __init__(
        self,
        model: PyTorchModel,
        gp_input_names: List[str] = [],
        gp_outcome_names: List[str] = [],
    ):
        """Custom prior mean for a GP based on an arbitrary model.

        Args:
            model: Representation of the model.
            gp_input_names (List[str]): list of feature names in the order they
                are passed to the GP model
            gp_output_names (List[str]): list of outcome names in the order the
                GP model expects
        """
        super().__init__()
        self._model = model
        self._gp_input_names = gp_input_names
        self._gp_outcome_names = gp_outcome_names

    def evaluate_model(self, x: Dict[str, torch.Tensor]):
        """Placeholder method which can be used to modify model calls."""
        return self._model.evaluate(x)

    def manipulate_outcome(self, y_model: Dict[str, torch.Tensor]):
        """Placeholder method which can be used to modify the outcome
        of the model calls, e.g. adding extra outputs"""
        return y_model

    def forward(self, x: torch.Tensor):
        # incoming tensor will be of the shape [b,n,m] where b is the batch
        # number, n is the number of samples and m is the number of features
        # we need to break up this tensor into a dictionary format that the
        # PyTorchModel will accept
        model_input = self._tensor_to_dictionary(x)
        # evaluate model
        y_model = self.evaluate_model(model_input)
        y_model = self.manipulate_outcome(y_model)
        # then once we have the model output in dictionary format we convert
        # it back to a tensor for botorch
        y = self._dictionary_to_tensor(y_model).squeeze()
        return y

    def _tensor_to_dictionary(self, x: torch.Tensor):
        input_dict = {}
        for idx, feature in enumerate(self._gp_input_names):
            input_dict[feature] = x[..., idx].unsqueeze(
                -1
            )  # index by the last dimension
        return input_dict

    def _dictionary_to_tensor(self, y_model: Dict[str, torch.Tensor]):
        output_tensor = torch.stack(
            [y_model[outcome] for outcome in self._gp_outcome_names]
        )
        return output_tensor.squeeze()
