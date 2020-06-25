from lume_model.models import SurrogateModel
from lume_model.variables import ScalarOutputVariable, ScalarInputVariable
import pytest


def test_surrogate_model_construction():
    class ExampleModel(SurrogateModel):
        input_variables = {
            "input1": ScalarInputVariable(name="input1", value=1, range=[0.0, 5.0]),
            "input2": ScalarInputVariable(name="input2", value=2, range=[0.0, 5.0]),
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
            return list(self.output_variables.values())

    ExampleModel()

    class ExampleFailureModel(SurrogateModel):
        input_variables = {
            "input1": ScalarInputVariable(name="input1", value=1, range=[0.0, 5.0]),
            "input2": ScalarInputVariable(name="input2", value=2, range=[0.0, 5.0]),
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
            return list(self.output_variables.values())

    # now a failure:

    with pytest.raises(TypeError):
        ExampleFailureModel()
