from lume_model.models import BaseModel
from lume_model.variables import ScalarOutputVariable, ScalarInputVariable
import pytest


def test_surrogate_model_construction():
    class ExampleModel(BaseModel):
        input_variables = {
            "input1": ScalarInputVariable(name="input1", default=1, range=[0.0, 5.0]),
            "input2": ScalarInputVariable(name="input2", default=2, range=[0.0, 5.0]),
        }

        output_variables = {
            "output1": ScalarOutputVariable(name="output1"),
            "output2": ScalarOutputVariable(name="output2"),
        }

        def evaluate(self, input_variables):

            self.input_variables = {
                variable.name: variable for variable in input_variables
            }

            self.output_variables["output1"].value = (
                self.input_variables["input1"].value * 2
            )
            self.output_variables["output2"].value = (
                self.input_variables["input2"].value * 2
            )

            # return inputs * 2
            return self.output_variables

    ExampleModel()

    class ExampleFailureModel(BaseModel):
        input_variables = {
            "input1": ScalarInputVariable(name="input1", default=1, range=[0.0, 5.0]),
            "input2": ScalarInputVariable(name="input2", default=2, range=[0.0, 5.0]),
        }

        output_variables = {
            "output1": ScalarOutputVariable(name="output1"),
            "output2": ScalarOutputVariable(name="output2"),
        }

        def predict(self, input_variables):

            self.input_variables = {
                variable.name: variable for variable in input_variables
            }

            self.output_variables["output1"].value = (
                self.input_variables["input1"].value * 2
            )
            self.output_variables["output2"].value = (
                self.input_variables["input2"].value * 2
            )

            # return inputs * 2
            return self.output_variables

    # now a failure:

    with pytest.raises(TypeError):
        ExampleFailureModel()
