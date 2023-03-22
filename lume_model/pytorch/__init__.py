import logging
from typing import Dict, Optional, List

import torch

from lume_model.models import BaseModel
from lume_model.variables import InputVariable, OutputVariable
from botorch.models.transforms.input import ReversibleInputTransform

logger = logging.getLogger(__name__)


class PyTorchModel(BaseModel):
    def __init__(
        self,
        model_file: str,
        input_variables: Dict[str, InputVariable],
        output_variables: Dict[str, OutputVariable],
        input_transformers: Optional[List[ReversibleInputTransform]] = [],
        output_transformers: Optional[List[ReversibleInputTransform]] = [],
        output_format: Optional[dict] = None,
        feature_order: Optional[list] = None,
        output_order: Optional[list] = None,
    ) -> None:
        """Initializes the model and stores inputs/outputs.

        Args:
            model_file (str): Path to model file generated with torch.save()
            input_variables (List[InputVariable]): list of model input variables
            output_variables (List[OutputVariable]): list of model output variables
            input_transformers: (List[ReversibleInputTransform]): list of transformer
                objects to apply to input before passing to model
            output_transformers: (List[ReversibleInputTransform]): list of transformer
                objects to apply to output of model
            output_format (Optional[dict]): Wrapper for interpreting outputs. This now handles
                raw or softmax values, but should be expanded to accomodate misc
                functions. Now, dictionary should look like:
                    {"type": Literal["raw", "string"]}
            feature_order: List[str]: list containing the names of features in the
                order in which they are passed to the model
            output_order: List[str]: list containing the names of outputs in the
                order the model produces them

        TODO: make list of Transformer objects into botorch ChainedInputTransform?

        """
        super(BaseModel, self).__init__()

        # Save init
        self.input_variables = input_variables
        self.output_variables = output_variables
        self._model_file = model_file
        self._output_format = output_format

        # make sure all of the transformers are in eval mode
        self._input_transformers = input_transformers
        for transformer in self._input_transformers:
            transformer.eval()
        self._output_transformers = output_transformers
        for transformer in self._output_transformers:
            # this swaps the transform/untransform so the
            # forward pass is now the inverse transformation
            # which allows us to use the input->model scaling
            # factors
            transformer.reverse = True
            transformer.eval()

        self._model = torch.load(model_file)

        self._feature_order = feature_order
        self._output_order = output_order

    @property
    def features(self):
        return self._feature_order

    @property
    def outputs(self):
        return self._output_order

    def evaluate(
        self, input_variables: Dict[str, InputVariable], return_raw: bool = False
    ) -> Dict[str, OutputVariable]:
        """Evaluate model using new input variables.

        Args:
            input_variables (Dict[str, InputVariable]): List of updated input variables
            return_raw (bool): flag to determine whether the dictionary returned
                contains raw float/int values or OutputVariable objects

        Returns:
            Dict[str, OutputVariable]: List of output variables

        """
        # all PyTorch models will follow the same process, the inputs
        # are formatted, then converted to model features. Then they
        # are passed through the model, and transformed again on the
        # other side. The final dictionary is then converted into a
        # useful form
        input_vals = self._prepare_inputs(input_variables)
        input_vals = self._arrange_inputs(input_vals)
        features = self._transform_inputs(input_vals)
        raw_output = self._model(features)
        transformed_output = self._transform_outputs(raw_output)
        output = self._parse_outputs(transformed_output)
        output = self._prepare_outputs(output, return_raw)

        return output

    def _prepare_inputs(
        self, input_variables: Dict[str, InputVariable]
    ) -> Dict[str, float]:
        """
        Prepares the input variables dictionary as a format appropriate
        to be passed to the transformers and updates the stored InputVariables
        with new values

        Args:
            input_variables (dict): Dictionary of input variable names to
                variables in any format (InputVariable or raw values)

        Returns:
            dict (Dict[str, (float,int)]): dictionary of input variable values
                to be passed to the transformers
        """
        for var_name, var in self.input_variables.items():
            try:
                if isinstance(input_variables[var_name], InputVariable):
                    var.value = input_variables[var_name].value
                else:
                    var.value = input_variables[var_name]
            except KeyError as e:
                logger.warning(f"{e} missing from input_dict, using default value")
                var.value = var.default

        return {var_name: var.value for var_name, var in self.input_variables.items()}

    def _arrange_inputs(self, input_variables: Dict[str, float]) -> torch.Tensor:
        """
        Enforces the order of the input variables to be passed to the transformers
        and models

        Args:
            input_variables (dict): Dictionary of input variable names to raw
                values of inputs

        Returns:
            torch.Tensor: ordered tensor of input variables to be passed to the
                transformers

        """
        features = []
        if self._feature_order is not None:
            for feature_name in self._feature_order:
                features.append(input_variables[feature_name])
        else:
            # if there's no order specified, we assume it's the same as the
            # order passed in the variables.yml file
            for feature_name in self.input_variables.keys():
                features.append(input_variables[feature_name])

        return torch.Tensor(features)

    def _transform_inputs(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to the inputs

        Args:
            input_values (torch.Tensor): tensor of input variables to be passed
                to the transformers

        Returns:
            torch.Tensor: tensor of transformed input variables to be passed
                to the model
        """
        for transformer in self._input_transformers:
            input_values = transformer(input_values)
        return input_values

    def _transform_outputs(self, model_output: torch.Tensor) -> torch.Tensor:
        """
        Untransforms the model outputs to real units

        Args:
            model_output (torch.Tensor): tensor of outputs from the model

        Returns:
            Dict[str, torch.Tensor]: dictionary of variable name to tensor
                of untransformed output variables
        """
        # NOTE do we need to sort these to reverse them?
        for transformer in self._output_transformers:
            model_output = transformer(model_output)
        return model_output

    def _parse_outputs(self, model_output: torch.Tensor) -> Dict[str, float]:
        """
        Constructs dictionary from model outputs

        Args:
            model_output (torch.Tensor): transformed output from NN model

        Returns:
            Dict[str, float]: dictionary of output variable name to output
                value
        """
        output = {}
        if self._output_order is not None:
            for idx, output_name in enumerate(self._output_order):
                output[output_name] = model_output[idx].detach().item()
        else:
            # if there's no order specified, we assume it's the same as the
            # order passed in the variables.yml file
            for idx, output_name in enumerate(self.output_variables.keys()):
                output[output_name] = model_output[idx].detach().item()
        return output

    def _prepare_outputs(
        self, predicted_output: dict, return_raw
    ) -> Dict[str, OutputVariable]:
        """
        Converts the
        Args:
            predicted_output (Dict[str, float]): Dictionary of output variable name to
                value

        Returns:
            Dict[str, OutputVariable]: Dictionary of output variable name to output
                variable.
        """
        for variable in self.output_variables.values():
            if variable.variable_type == "scalar":
                self.output_variables[variable.name].value = predicted_output[
                    variable.name
                ]

            elif variable.variable_type == "image":
                self.output_variables[variable.name].value = predicted_output[
                    variable.name
                ].reshape(variable.shape)

                # update limits
                if self.output_variables[variable.name].x_min_variable:
                    self.output_variables[variable.name].x_min = predicted_output[
                        self.output_variables[variable.name].x_min_variable
                    ]

                if self.output_variables[variable.name].x_max_variable:
                    self.output_variables[variable.name].x_max = predicted_output[
                        self.output_variables[variable.name].x_max_variable
                    ]

                if self.output_variables[variable.name].y_min_variable:
                    self.output_variables[variable.name].y_min = predicted_output[
                        self.output_variables[variable.name].y_min_variable
                    ]

                if self.output_variables[variable.name].y_max_variable:
                    self.output_variables[variable.name].y_max = predicted_output[
                        self.output_variables[variable.name].y_max_variable
                    ]
        if return_raw:
            return {key: var.value for key, var in self.output_variables.items()}
        else:
            return self.output_variables
