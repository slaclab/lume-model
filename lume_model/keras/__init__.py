import copy
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import List
import logging
from tensorflow.keras.models import load_model

from lume_model.models import SurrogateModel
from lume_model.utils import load_variables
from lume_model.variables import InputVariable, OutputVariable
from lume_model.keras.layers import ScaleLayer, UnscaleLayer, UnscaleImgLayer

logger = logging.getLogger(__name__)


class BaseModel(SurrogateModel, ABC):
    """
    The BaseModel class is used for the loading and evaluation of online models. It is an abstract base class designed to
    implement the general behaviors expected for models used with the Keras lume-model tool kit. Parsing methods for inputs and
    outputs must be implemented in derived classes.

    Attributes:


    """

    def __init__(
        self,
        model_file: str,
        input_variables: List[InputVariable],
        output_variables: List[OutputVariable],
        input_format: dict = None,
        output_format: dict = None,
    ) -> None:
        """Initializes the model and stores inputs/outputs.

        Args:
            model_file (str): Path to model file generated with keras.save()
            input_variables (List[InputVariable]): list of model input variables
            output_variables (List[OutputVariable]): list of model output variables
            _thread_graph (tf.Graph): default graph for model execution

        """

        # Save init
        self.model_file = model_file
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.input_format = input_format
        self.output_format = output_format

        # load model in thread safe manner
        self._thread_graph = tf.Graph()
        with self._thread_graph.as_default():
            self.model = load_model(
                model_file,
                custom_objects={
                    "ScaleLayer": ScaleLayer,
                    "UnscaleLayer": UnscaleLayer,
                    "UnscaleImgLayer": UnscaleImgLayer,
                },
            )

    def evaluate(self, input_variables: List[InputVariable]) -> List[OutputVariable]:
        """Evaluate model using new input variables.

        Args:
            input_variables (List[InputVariable]): List of updated input variables

        Returns:
            List[OutputVariable]: List of output variables

        """
        self.input_variables = {var.name: var for var in input_variables}

        # convert list of input variables to dictionary
        input_dictionary = {
            input_variable.name: input_variable.value
            for input_variable in input_variables
        }

        # MUST IMPLEMENT A format_input METHOD TO CONVERT FROM DICT -> MODEL INPUT
        formatted_input = self.format_input(input_dictionary)

        # call prediction in threadsafe manner
        with self._thread_graph.as_default():
            model_output = self.model.predict(formatted_input)

        # MUST IMPLEMENT AN OUTPUT -> DICT METHOD
        output = self.parse_output(model_output)

        # PREPARE OUTPUTS WILL FORMAT RETURN VARIABLES (DICT-> VARIABLES)
        return self.prepare_outputs(output)

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

    def prepare_outputs(self, predicted_output: dict):
        """Prepares the model outputs to be served so that no additional manipulation
        occurs in the OnlineSurrogateModel class.

        Args:
            model_outputs (dict): Dictionary of output variables to np.ndarrays of outputs

        Returns:
            dict: Dictionary of output variables to respective scalars
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

        return list(self.output_variables.values())

    def format_input(self, input_dictionary: dict):
        """Formats input to be fed into model

        Args:
            input_dictionary (dict): Dictionary mapping input to value.
        """

        vector = []
        for item in self.input_format["order"]:
            vector.append(input_dictionary[item])

        # Convert to numpy array and reshape
        vector = np.array(vector)
        vector = vector.reshape(tuple(self.input_format["shape"]))

        return vector

    @abstractmethod
    def parse_output(self, model_output):
        # MUST IMPLEMENT A METHOD TO CONVERT MODEL OUTPUT TO A DICTIONARY OF VARIABLE NAME -> VALUE
        pass
