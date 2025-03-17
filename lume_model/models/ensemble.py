import torch
from torch.distributions import Normal
from torch.distributions.distribution import Distribution as TDistribution

from lume_model.models.prob_model_base import ProbModelBaseModel
from lume_model.models.torch_model import TorchModel


class NNEnsemble(ProbModelBaseModel):
    """LUME-model class for neural network ensembles.

    This class allows for the evaluation of multiple NN models or a
    single probabilistic NN model with multiple predictions.

    Args:
        models: List of one or more LUME-model neural network models.
    """

    models: list[TorchModel]

    # for a list of models, each will return a dict of output names with the value
    # so we need to take mean and var of the output values for each output name
    # for a single model, it will return a dict of output names with the values,
    # it will either be "mean" and "var" with float values, or "output1" etc
    # with tensor values that we'd need to take mean and var of
    # TODO: will single model return mean and var for each output name? is that compatible
    #       with the current implementation of TorchModel?
    # TODO: would single model be a ProbModelBaseModel?

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_predictions(
        self, input_dict: dict[str, float | torch.Tensor]
    ) -> dict[str, TDistribution]:
        """Get the predictions of the ensemble of models.

        Args:
            x: Input tensor.

        Returns:
            Distribution of the predictions.
        """
        # TorchModels take a dict of input variable names to values
        # so no need to convert the input to tensor
        predictions = []
        for model in self.models:
            predictions.append(model.evaluate(input_dict))
        # Return a dictionary of output variable names to distributions
        return self._create_output_dict(predictions)

    def _create_output_dict(self, output_list: list) -> dict[str, TDistribution]:
        """Creates the output dictionary from the distribution.

        Args:
            output_list: List of output dictionaries.

        Returns:
            Dictionary of output variable names to distributions.
        """

        # Predictions are a list of dicts of output names to values
        # need to map them to a dict of output names to distributions
        ensemble_output_dict = {}

        for key in output_list[0]:  # for each named output
            output_tensor = torch.tensor([d[key].tolist() for d in output_list])
            # TODO: use whatever distribution is defined for that particular output
            #       in self.output_variables["distribution_type"]
            ensemble_output_dict[key] = Normal(
                output_tensor.mean(axis=0),
                torch.sqrt(output_tensor.var(axis=0)),
            )
        return ensemble_output_dict

    @property
    def _tkwargs(self):
        """Returns the device and dtype for the model."""
        return {"device": self.device, "dtype": self.dtype}
