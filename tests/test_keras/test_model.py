import os
import random
from typing import Union
from copy import deepcopy

import pytest
import numpy as np

try:
    from lume_model.keras import KerasModel
    from lume_model.variables import InputVariable, OutputVariable, ScalarOutputVariable
except ImportError:
    pass


# def assert_variables_updated(
#     input_value: float,
#     output_value: float,
#     model,
#     input_name: str,
#     output_name: str,
# ):
#     """helper function to verify that model input_variables and output_variables
#     have been updated correctly with float values (NOT arrays)"""
#     assert isinstance(model.input_variables[model.input_names.index(input_name)].value, float)
#     assert model.input_variables[model.input_names.index(input_name)].value == pytest.approx(input_value)
#     assert isinstance(model.output_variables[model.output_names.index(output_name)].value, int)
#     assert model.output_variables[model.output_names.index(output_name)].value == pytest.approx(output_value)


# def assert_iris_model_result(iris_test_input_dict: dict, model):
#     assert_variables_updated(
#         input_value=iris_test_input_dict["SepalLength"].item(),
#         output_value=2,
#         model=model,
#         input_name="SepalLength",
#         output_name="Species",
#     )


def assert_model_equality(m1: KerasModel, m2: KerasModel):
    assert m1.input_variables == m2.input_variables
    assert m1.output_variables == m2.output_variables
    for l1, l2 in zip(m1.model.layers, m2.model.layers):
        assert l1.get_config() == l2.get_config()
    for l1, l2 in zip(m1.model.layers, m2.model.layers):
        w1, w2 = l1.get_weights(), l2.get_weights()
        assert len(w1) == len(w2)
        for i, w in enumerate(w1):
            assert np.array_equal(w, w2[i])
    assert m1.output_format == m2.output_format
    assert m1.output_transforms == m2.output_transforms


class TestKerasModel:
    def test_model_from_objects(
            self,
            iris_model_kwargs: dict[str, Union[list, dict, str]],
            iris_variables: tuple[list[InputVariable], list[OutputVariable]],
            iris_model,
    ):
        input_variables, output_variables = iris_variables

        assert isinstance(iris_model, KerasModel)
        assert iris_model.input_variables == input_variables
        assert iris_model.output_variables == output_variables

    def test_model_from_yaml(self, rootdir: str, iris_model):
        file = f"{rootdir}/test_files/iris_classification/keras_model.yml"
        yaml_model = KerasModel(file)
        assert_model_equality(yaml_model, iris_model)

    def test_model_as_yaml(self, rootdir: str, iris_model):
        filename = "test_keras_model"
        file = f"{filename}.yml"
        iris_model.dump(file)
        yaml_model = KerasModel(file)
        assert_model_equality(yaml_model, iris_model)
        os.remove(file)
        os.remove(f"{filename}_model.keras")

    def test_model_evaluate_variable(
            self,
            iris_test_input_dict: dict,
            iris_model_kwargs: dict[str, Union[list, dict, str]],
    ):
        kwargs = {k: v if not k == "output_format" else "variable" for k, v in iris_model_kwargs.items()}
        iris_model = KerasModel(**kwargs)
        input_variables = deepcopy(iris_model.input_variables)
        for var in input_variables:
            var.value = iris_test_input_dict[var.name].item()
        results = iris_model.evaluate({var.name: var for var in input_variables})

        assert isinstance(results["Species"], ScalarOutputVariable)
        assert results["Species"].value == 2
        # assert_iris_model_result(iris_test_input_dict, iris_model)

    def test_model_evaluate_single_sample(self, iris_test_input_dict: dict, iris_model):
        results = iris_model.evaluate(iris_test_input_dict)

        assert isinstance(results["Species"], np.ndarray)
        assert results["Species"] == 2
        # assert_iris_model_result(iris_test_input_dict, iris_model)

    def test_model_evaluate_n_samples(self, iris_test_input_array, iris_model):
        test_dict = {
            key: iris_test_input_array[:, idx] for idx, key in enumerate(iris_model.input_names)
        }
        results = iris_model.evaluate(test_dict)
        # in this case we don't expect the input/output variables to be updated, because we don't know which value
        # to update them with so we only check for the resulting values
        target_array = np.array([2, 0, 1], dtype=results["Species"].dtype)

        assert all(np.isclose(results["Species"], target_array))

    # def test_model_evaluate_batch_n_samples(
    #         self,
    #         iris_test_input_array,
    #         iris_model,
    # ):
    #     # model should be able to handle input of shape [batch_size, n_samples, n_dim]
    #     input_dict = {
    #         key: iris_test_input_array.reshape((1, *iris_test_input_array.shape)).repeat(2, axis=0)
    #         for idx, key in enumerate(iris_model.input_names)
    #     }
    #     results = iris_model.evaluate(input_dict)
    #
    #     # output shape should be [batch_size, n_samples]
    #     assert tuple(results["Species"].shape) == (2, 3)

    def test_model_evaluate_raw(
            self,
            iris_test_input_dict: dict,
            iris_model_kwargs: dict[str, Union[list, dict, str]],
    ):
        kwargs = {k: v if not k == "output_format" else "raw" for k, v in iris_model_kwargs.items()}
        iris_model = KerasModel(**kwargs)
        float_dict = {key: value.item() for key, value in iris_test_input_dict.items()}
        results = iris_model.evaluate(float_dict)

        assert isinstance(results["Species"], int)
        assert results["Species"] == 2
        # assert_iris_model_result(iris_test_input_dict, iris_model)

    def test_model_evaluate_shuffled_input(self, iris_test_input_dict: dict, iris_model):
        shuffled_input = deepcopy(iris_test_input_dict)
        item_list = list(shuffled_input.items())
        random.shuffle(item_list)
        shuffled_input = dict(item_list)
        results = iris_model.evaluate(shuffled_input)

        assert isinstance(results["Species"], np.ndarray)
        assert results["Species"] == 2
        # assert_iris_model_result(iris_test_input_dict, iris_model)

    @pytest.mark.parametrize("test_idx,expected", [(0, 2), (1, 0), (2, 1)])
    def test_model_evaluate_different_values(
            self,
            test_idx: int,
            expected: float,
            iris_test_input_array,
            iris_model,
    ):
        input_dict = {
            key: iris_test_input_array[test_idx, idx] for idx, key in enumerate(iris_model.input_names)
        }
        results = iris_model.evaluate(input_dict)

        assert results["Species"].item() == expected
        # assert_variables_updated(
        #     input_value=input_dict["SepalWidth"].item(),
        #     output_value=expected,
        #     model=iris_model,
        #     input_name="SepalWidth",
        #     output_name="Species",
        # )
