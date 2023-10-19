import os

import pytest

from lume_model.utils import verify_unique_variable_names, variables_as_yaml, variables_from_yaml


def test_verify_unique_variable_names(simple_variables):
    input_variables = simple_variables["input_variables"]
    output_variables = simple_variables["output_variables"]
    # unique variables names
    verify_unique_variable_names(input_variables)
    verify_unique_variable_names(output_variables)
    # non-unique input names
    original_name = input_variables[1].name
    input_variables[1].name = input_variables[0].name
    with pytest.raises(ValueError):
        verify_unique_variable_names(input_variables)
    input_variables[1].name = original_name
    # non-unique output names
    original_name = output_variables[1].name
    output_variables[1].name = output_variables[0].name
    with pytest.raises(ValueError):
        verify_unique_variable_names(output_variables)
    output_variables[1].name = original_name


def test_variables_as_yaml(simple_variables):
    file = "test_variables.yml"
    variables_as_yaml(**simple_variables, file=file)
    os.remove(file)


def test_variables_as_and_from_yaml(simple_variables):
    file = "test_variables.yml"
    variables_as_yaml(**simple_variables, file=file)
    variables = variables_from_yaml(file)
    os.remove(file)
    assert simple_variables["input_variables"] == variables[0]
    assert simple_variables["output_variables"] == variables[1]
