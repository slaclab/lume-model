from lume_model import utils
import os
import sys
import pytest
from lume_model.models import SurrogateModel
from lume_model.variables import (
    InputVariable,
    OutputVariable,
    ScalarInputVariable,
    ScalarOutputVariable,
)


def test_save():
    input_variables = {
        "input1": ScalarInputVariable(name="input1", default=1, range=[0.0, 5.0]),
        "input2": ScalarInputVariable(name="input2", default=2, range=[0.0, 5.0]),
    }

    output_variables = {
        "output1": ScalarOutputVariable(name="output1"),
        "output2": ScalarOutputVariable(name="output2"),
    }

    file_name = "test_variables.pickle"
    utils.save_variables(input_variables, output_variables, file_name)
    utils.load_variables(file_name)
    os.remove(file_name)


def test_variables_with_same_name():
    input_variables = {
        "input1": ScalarInputVariable(name="input1", default=1, range=[0.0, 5.0]),
        "input2": ScalarInputVariable(name="input2", default=2, range=[0.0, 5.0]),
    }

    output_variables = {
        "input1": ScalarOutputVariable(name="input1"),
        "output2": ScalarOutputVariable(name="output2"),
    }

    file_name = "test_variables.pickle"

    with pytest.raises(ValueError):
        utils.save_variables(input_variables, output_variables, file_name)
