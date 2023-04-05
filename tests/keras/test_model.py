import pytest
import sys
from lume_model import utils
from lume_model.variables import InputVariable, OutputVariable

try:
    from lume_model.keras import KerasModel
except ImportError:
    pass


def test_model_from_yaml(config_file):
    pytest.importorskip("tensorflow", minversion="2.3.1")
    model = utils.model_from_yaml(config_file)
    assert isinstance(model, KerasModel)


def test_variables_from_yaml(config_file):
    input_variables, output_variables = utils.variables_from_yaml(config_file)

    for variable_name, variable in input_variables.items():
        assert isinstance(variable, InputVariable)

    for variable_name, variable in output_variables.items():
        assert isinstance(variable, OutputVariable)
