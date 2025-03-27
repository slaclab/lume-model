import warnings
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
            # TODO: use whatever distribution is defined for that particular output
            #       in self.output_variables["distribution_type"], if none, take Normal
            ensemble_output_dict[key] = Normal(
                output_tensor.mean(axis=0),
                torch.sqrt(output_tensor.var(axis=0)),
            )
        return ensemble_output_dict

    @property
    def _tkwargs(self):
        """Returns the device and dtype for the model."""
        return {"device": self.device, "dtype": self.dtype}
