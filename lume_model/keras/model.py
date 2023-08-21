import os
import logging
from typing import Union

import keras
import numpy as np
from pydantic import validator

from lume_model.base import LUMEBaseModel
from lume_model.variables import (
    InputVariable,
    OutputVariable,
    ScalarInputVariable,
    ScalarOutputVariable,
    ImageOutputVariable,
)

logger = logging.getLogger(__name__)


class KerasModel(LUMEBaseModel):
    """LUME-model class for keras models.

    Attributes:
        model: The keras base model.
        output_format: Determines format of outputs: "array", "variable" or "raw".
        output_transforms: List of strings defining additional transformations applied to the outputs. For now,
          only "softmax" is supported.
    """
    model: keras.Model
    output_format: str = "array"
    output_transforms: list[str] = []

    def __init__(
            self,
            config: Union[dict, str] = None,
            **kwargs,
    ):
        """Initializes KerasModel.

        Args:
            config: Model configuration as dictionary, YAML or JSON formatted string or file path. This overrides
              all other arguments.
            **kwargs: See class attributes.
        """
        super().__init__(config, **kwargs)

    @validator("model", pre=True)
    def validate_keras_model(cls, v):
        if isinstance(v, (str, os.PathLike)):
            if os.path.exists(v):
                v = keras.models.load_model(v)
        return v

    @validator("output_format")
    def validate_output_format(cls, v):
        supported_formats = ["array", "variable", "raw"]
        if v not in supported_formats:
            raise ValueError(f"Unknown output format {v}, expected one of {supported_formats}.")
        return v

    @property
    def dtype(self):
        return np.double

    def evaluate(
            self,
            input_dict: dict[str, Union[InputVariable, float, np.ndarray]],
    ) -> dict[str, Union[OutputVariable, float, np.ndarray]]:
        """Evaluates model on the given input dictionary.

        Args:
            input_dict: Input dictionary on which to evaluate the model.

        Returns:
            Dictionary of output variable names to values.
        """
        formatted_inputs = self._format_inputs(input_dict)
        complete_input_dict = self._complete_inputs(formatted_inputs)
        output_array = self.model.predict(complete_input_dict).astype(self.dtype)
        output_array = self._output_transform(output_array)
        parsed_outputs = self._parse_outputs(output_array)
        output_dict = self._prepare_outputs(parsed_outputs)
        return output_dict

    def random_input(self, n_samples: int = 1) -> dict[str, np.ndarray]:
        """Generates random input(s) for the model.

        Args:
            n_samples: Number of random samples to generate.

        Returns:
            Dictionary of input variable names to arrays.
        """
        input_dict = {}
        for var in self.input_variables:
            if isinstance(var, ScalarInputVariable):
                input_dict[var.name] = np.random.uniform(*var.value_range, size=n_samples)
            else:
                default_array = np.array(var.default, dtype=self.dtype)
                input_dict[var.name] = np.repeat(default_array.reshape((1, *default_array.shape)),
                                                 n_samples, axis=0)
        return input_dict

    def random_evaluate(self, n_samples: int = 1) -> dict[str, Union[OutputVariable, float, np.ndarray]]:
        """Returns random evaluation(s) of the model.

        Args:
            n_samples: Number of random samples to evaluate.

        Returns:
            Dictionary of variable names to outputs.
        """
        random_input = self.random_input(n_samples)
        return self.evaluate(random_input)

    def _format_inputs(
            self,
            input_dict: dict[str, Union[InputVariable, float, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Formats values of the input dictionary as arrays.

        Args:
            input_dict: Dictionary of input variable names to values.

        Returns:
            Dictionary of input variable names to arrays.
        """
        # NOTE: The input variable is only updated if a singular value is given (ambiguous otherwise)
        formatted_inputs = {}
        for var_name, var in input_dict.items():
            if isinstance(var, InputVariable):
                formatted_inputs[var_name] = np.array(var.value, dtype=self.dtype)
                # self.input_variables[self.input_names.index(var_name)].value = var.value
            elif isinstance(var, float):
                formatted_inputs[var_name] = np.array(var, dtype=self.dtype)
                # self.input_variables[self.input_names.index(var_name)].value = var
            elif isinstance(var, np.ndarray):
                var = var.astype(self.dtype).squeeze()
                formatted_inputs[var_name] = var
                # if var.ndim == 0:
                #     self.input_variables[self.input_names.index(var_name)].value = var.item()
            else:
                TypeError(
                    f"Unknown type {type(var)} passed to evaluate."
                    f"Should be one of InputVariable, float or np.ndarray."
                )
        return formatted_inputs

    def _complete_inputs(self, formatted_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Completes input dictionary by filling in default values.

        Args:
            formatted_inputs: Dictionary of input variable names to arrays.

        Returns:
            Completed input dictionary to be passed to the model.
        """
        # determine input shape
        input_shapes = [formatted_inputs[k].shape for k in formatted_inputs.keys()]
        if not all(ele == input_shapes[0] for ele in input_shapes):
            raise ValueError("Inputs have inconsistent shapes.")

        for i, key in enumerate(self.input_names):
            if key not in formatted_inputs.keys():
                default_array = np.array(self.input_variables[i].default, dtype=self.dtype)
                formatted_inputs[key] = np.tile(default_array, reps=input_shapes[0])

        if not input_shapes[0]:
            for key in self.input_names:
                formatted_inputs[key] = formatted_inputs[key].reshape((1, *formatted_inputs[key].shape))
        return formatted_inputs

    def _output_transform(self, output_array: np.ndarray) -> np.ndarray:
        """Applies additional transformations to the model output array.

        Args:
            output_array: Output array from the model.

        Returns:
            Transformed output array.
        """
        if "softmax" in self.output_transforms:
            output_array = np.argmax(output_array, axis=-1)
        return output_array

    def _parse_outputs(self, output_array: np.ndarray) -> dict[str, np.ndarray]:
        """Constructs dictionary from model output array.

        Args:
            output_array: Transformed output array from the model.

        Returns:
            Dictionary of output variable names to transformed arrays.
        """
        parsed_outputs = {}
        if output_array.ndim in [0, 1]:
            output_array = output_array.reshape((1, *output_array.shape))
        if len(self.output_names) == 1:
            parsed_outputs[self.output_names[0]] = output_array.squeeze()
        else:
            for idx, output_name in enumerate(self.output_names):
                parsed_outputs[output_name] = output_array[..., idx].squeeze()
        return parsed_outputs

    def _prepare_outputs(
            self,
            parsed_outputs: dict[str, np.ndarray],
    ) -> dict[str, Union[OutputVariable, np.ndarray]]:
        """Updates and returns outputs according to output_format.

        Updates the output variables within the model to reflect the new values.

        Args:
            parsed_outputs: Dictionary of output variable names to transformed arrays.

        Returns:
            Dictionary of output variable names to values depending on output_format.
        """
        # for var in self.output_variables:
        #     if parsed_outputs[var.name].ndim == 0:
        #         idx = self.output_names.index(var.name)
        #         if isinstance(var, ScalarOutputVariable):
        #             self.output_variables[idx].value = parsed_outputs[var.name].item()
        #         elif isinstance(var, ImageOutputVariable):
        #             # OutputVariables should be arrays
        #             self.output_variables[idx].value = (parsed_outputs[var.name].reshape(var.shape).numpy())
        #             self._update_image_limits(var, parsed_outputs)

        if self.output_format == "array":
            return parsed_outputs
        elif self.output_format == "variable":
            output_dict = {var.name: var for var in self.output_variables}
            for var in output_dict.values():
                var.value = parsed_outputs[var.name].item()
            return output_dict
            # return {var.name: var for var in self.output_variables}
        else:
            return {key: value.item() if value.squeeze().ndim == 0 else value
                    for key, value in parsed_outputs.items()}
            # return {var.name: var.value for var in self.output_variables}

    def _update_image_limits(
            self,
            variable: OutputVariable, predicted_output: dict[str, np.ndarray],
    ):
        output_idx = self.output_names.index(variable.name)
        if self.output_variables[output_idx].x_min_variable:
            self.output_variables[output_idx].x_min = predicted_output[
                self.output_variables[output_idx].x_min_variable
            ].item()

        if self.output_variables[output_idx].x_max_variable:
            self.output_variables[output_idx].x_max = predicted_output[
                self.output_variables[output_idx].x_max_variable
            ].item()

        if self.output_variables[output_idx].y_min_variable:
            self.output_variables[output_idx].y_min = predicted_output[
                self.output_variables[output_idx].y_min_variable
            ].item()

        if self.output_variables[output_idx].y_max_variable:
            self.output_variables[output_idx].y_max = predicted_output[
                self.output_variables[output_idx].y_max_variable
            ].item()
