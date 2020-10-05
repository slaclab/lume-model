import pytest
from lume_model import utils
from lume_model.variables import InputVariable, OutputVariable


@pytest.mark.skipif(
    "tensorflow" not in sys.modules, reason="requires tensorflow installation"
)
def test_model_from_yaml(config_file):
    model = utils.model_from_yaml(config_file)
    model.random_evaluate()


@pytest.mark.skipif(
    "tensorflow" not in sys.modules, reason="requires tensorflow installation"
)
def test_variables_from_yaml(config_file):
    input_variables, output_variables = utils.variables_from_yaml(config_file)

    for variable_name, variable in input_variables.items():
        assert isinstance(variable, InputVariable)

    for variable_name, variable in output_variables.items():
        assert isinstance(variable, OutputVariable)
