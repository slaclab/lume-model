"""
Adaptation of tensorflow tutorial: https://www.tensorflow.org/tutorials/estimator/premade
"""

from lume_model.keras import BaseModel
from lume_model.variables import ScalarInputVariable, ScalarOutputVariable
import numpy as np


class IrisModel(BaseModel):
    def format_input(self, input_dictionary):
        """Formats input to be fed into model
        """
        vector = np.array(
            [
                input_dictionary["SepalLength"],
                input_dictionary["SepalWidth"],
                input_dictionary["PetalLength"],
                input_dictionary["PetalWidth"],
            ]
        ).reshape(1, 4)

        return vector

    def parse_output(self, model_output):
        """Parses model output to create dictionary variable name -> value
        """
        softmax_output = list(model_output[0])
        return {"Species": softmax_output.index(max(softmax_output))}
