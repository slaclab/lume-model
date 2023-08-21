import os

import pytest

from lume_model.base import LUMEBaseModel


class ExampleModel(LUMEBaseModel):
    def evaluate(self, input_dict):
        pass


class TestBaseModel:
    def test_init(self, simple_variables):
        # init with no variable specification
        with pytest.raises(TypeError):
            _ = LUMEBaseModel()

        # init child class with no evaluate function
        class NoEvaluateModel(LUMEBaseModel):
            def predict(self, input_dict):
                pass

        with pytest.raises(TypeError):
            _ = NoEvaluateModel(**simple_variables)

        # init child class with evaluate function
        _ = ExampleModel(**simple_variables)

        # input and output variables sharing names is fine
        input_variables = simple_variables["input_variables"]
        output_variables = simple_variables["output_variables"]
        original_name = input_variables[0].name
        input_variables[0].name = output_variables[0].name
        _ = ExampleModel(**simple_variables)
        input_variables[0].name = original_name

    def test_serialization(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        file = "test_model.yml"
        _ = example_model.yaml(file)
        os.remove(file)

    def test_deserialization_from_config(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        file = "test_model.yml"
        _ = example_model.yaml(file)
        loaded_model = ExampleModel(file)
        os.remove(file)
        assert loaded_model.input_variables == example_model.input_variables
        assert loaded_model.output_variables == example_model.output_variables

    def test_input_names(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        for i, var in enumerate(simple_variables["input_variables"]):
            assert example_model.input_names.index(var.name) == i

    def test_output_names(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        for i, var in enumerate(simple_variables["output_variables"]):
            assert example_model.output_names.index(var.name) == i
