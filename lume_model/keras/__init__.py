import copy
import numpy as np
from typing import List, Dict, Optional, Any
import logging
from tensorflow.keras.models import load_model

from lume_model.models import BaseModel
from lume_model.variables import InputVariable, OutputVariable
from lume_model.keras.layers import ScaleLayer, UnscaleLayer, UnscaleImgLayer

logger = logging.getLogger(__name__)

base_layers = {
    "ScaleLayer": ScaleLayer,
    "UnscaleLayer": UnscaleLayer,
    "UnscaleImgLayer": UnscaleImgLayer,
}


class KerasModel(BaseModel):
    """
    The KerasModel class is used for the loading and evaluation of online models. It is  designed to
    implement the general behaviors expected for models used with the Keras lume-model tool kit.

    Attributes:
        input_valiables (Dict[str, InputVariable]): Dictionary mapping input variable name to variable
        output_variables (Dict[str, OutputVariable]): Dictionary mapping output variable name to variable
        _input_format (dict): Instructions for formatting model input
        _output_format (dict): Instructions for parsing model output

    """

    def __init__(
        self,
        model_file: str,
        input_variables: Dict[str, InputVariable],
        output_variables: Dict[str, OutputVariable],
        input_format: Optional[dict],
        output_format: Optional[dict],
        custom_layers: Optional[dict],
    ) -> None:
        """Initializes the model and stores inputs/outputs.

        Args:
            model_file (str): Path to model file generated with keras.save()
            input_variables (List[InputVariable]): list of model input variables
            output_variables (List[OutputVariable]): list of model output variables
            custom_layers (dict):

        """

        # Save init
        self.input_variables = input_variables
        self.output_variables = output_variables
        self._model_file = model_file
        self._input_format = input_format
        self._output_format = output_format

        base_layers.update(custom_layers)

        # load model in thread safe manner
        self._model = load_model(
            model_file,
            custom_objects=base_layers,
        )

    def evaluate(self, input_variables: Dict[str, InputVariable]) -> Dict[str, OutputVariable]:
        """Evaluate model using new input variables.

        Args:
            input_variables (Dict[str, InputVariable]): List of updated input variables

        Returns:
            Dict[str, OutputVariable]: List of output variables

        """
        self.input_variables = {var.name: var for var in input_variables}

        # convert list of input variables to dictionary
        input_dictionary = {
            input_variable.name: input_variable.value
            for input_variable in input_variables
        }

        # converts from input_dict -> formatted input
        formatted_input = self.format_input(input_dictionary)

        # call prediction in threadsafe manner
        model_output = self._model.predict(formatted_input)

        output = self.parse_output(model_output)

        # prepare outputs will format return variables (dict-> variables)
        return self._prepare_outputs(output)

    def random_evaluate(self) -> List[OutputVariable]:
        """Return a random evaluation of the model.

        Returns:
            List[OutputVariable]: List of outputs associated with random input

        """
        random_input = copy.deepcopy(self.input_variables)
        for variable in self.input_variables:
            if self.input_variables[variable].variable_type == "scalar":
                random_input[variable].value = np.random.uniform(
                    self.input_variables[variable].value_range[0],
                    self.input_variables[variable].value_range[1],
                )

            else:
                random_input[variable].value = self.input_variables[variable].default

        return self.evaluate(list(random_input.values()))

    def _prepare_outputs(
        self, predicted_output: Dict[str, Any]
    ) -> List[OutputVariable]:
        """Prepares the model outputs to be served so that no additional manipulation
        occurs in the BaseModel class.

        Args:
            model_outputs (dict): Dictionary of output variables to np.ndarrays of outputs

        Returns:
            List[OutputVariable]: List of output variables.
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

        return self.output_variables

    def format_input(
        self, input_dictionary: Dict[str, InputVariable]
    ) -> Dict[str, InputVariable]:
        """Formats input to be fed into model. For the base KerasModel, inputs should
        be assumed in dictionary format.

        Args:
            input_dictionary (Dict[str, InputVariable]): Dictionary mapping input to
                value.

        Returns:
            Dict[str, InputVariable]: Dictionary mapping variable name to formatted
                InputVariable.
        """
        formatted_dict = {}
        for input_variable, value in input_dictionary.items():
            if isinstance(value, (float, int)):
                formatted_dict[input_variable] = np.array([value])
            else:
                formatted_dict[input_variable] = [value]

        return formatted_dict

    def parse_output(self, model_output):
        """Parses model output to create dictionary variable name -> value. This assumes
        that outputs have been labeled during model creation.

        Args:
            model_output (np.ndarray): Raw model output

        Returns:
            Dict[str, OutputVariable]
        """
        output_dict = {}

        if not self._output_format.get("type") or self._output_format["type"] == "raw":
            for idx, output_name in enumerate(self._model.output_names):
                output_dict[output_name] = model_output[idx]

        elif self._output_format["type"] == "softmax":
            for idx, output_name in enumerate(self._model.output_names):
                softmax_output = list(model_output[idx])
                output_dict[output_name] = softmax_output.index(max(softmax_output))

        return output_dict
