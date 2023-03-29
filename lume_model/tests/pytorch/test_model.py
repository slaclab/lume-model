import logging
import random
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import pytest
import torch

from lume_model.pytorch import PyTorchModel
from lume_model.utils import model_from_yaml
from lume_model.variables import ScalarOutputVariable

"""
Things to Test:
---------------
- [x] we can load a PyTorch model from a yaml file
    - [x] returning the model class and keywords
    - [x] returning the model instance
- [x] we can create a PyTorch model from objects
- [x] the model correctly orders the features according to the
    specified feature order
- [x] pytorch model can be run using dictionary of:
    - [x] tensors (individual)
    - [x] tensors (of multiple samples for each feature)
    - [x] InputVariables
    - [x] floats
- [x] pytorch model evaluate() can return dictionary of either tensors
    or OutputVariables
- [x] pytorch model can be run with transformers or without
- [x] if we pass in a dictionary that's missing a value, we log
    an error and use the default value for the input
- [x] passing different input dictionaries through gives us different
    output dictionaries
- [x] differentiability through the model (required for Xopt)
- [ ] output transformations are applied in the correct order when we
    have multiple transformations
"""


def assert_variables_updated(
    input_value: float,
    output_value: float,
    model: PyTorchModel,
    input_name: str,
    output_name: str,
):
    """helper function to verify that model input_variables and output_variables
    have been updated correctly with float values (NOT tensors)"""
    assert isinstance(model.input_variables[input_name].value, float)
    assert model.input_variables[input_name].value == pytest.approx(input_value)
    assert isinstance(model.output_variables[output_name].value, float)
    assert model.output_variables[output_name].value == pytest.approx(output_value)


def test_model_from_yaml(
    rootdir: str, california_model_kwargs: Dict[str, Union[List, Dict, str]]
):
    with open(
        f"{rootdir}/test_files/california_regression/california_variables.yml",
        "r",
    ) as f:
        yaml_model, yaml_kwargs = model_from_yaml(f, load_model=False)

    assert yaml_model == PyTorchModel
    for key in list(california_model_kwargs.keys()):
        # we don't define anything about the transformers in the yml file so we
        # don't expect there to be anything in the california_model_kwargs about them
        if key not in ["input_transformers", "output_transformers"]:
            assert key in list(yaml_kwargs.keys())


def test_model_from_yaml_load_model(
    rootdir: str,
    california_variables: Tuple[dict, dict],
    california_transformers: Tuple[list, list],
    california_model_kwargs: Dict[str, Union[List, Dict, str]],
):
    input_variables, output_variables = california_variables
    input_transformer, output_transformer = california_transformers
    with open(
        f"{rootdir}/test_files/california_regression/california_variables.yml",
        "r",
    ) as f:
        yaml_model = model_from_yaml(f, load_model=True)

    assert isinstance(yaml_model, PyTorchModel)
    assert yaml_model.input_variables == input_variables
    assert yaml_model.output_variables == output_variables
    assert yaml_model.features == california_model_kwargs["feature_order"]
    assert yaml_model.outputs == california_model_kwargs["output_order"]
    assert yaml_model._input_transformers == []
    assert yaml_model._output_transformers == []

    # now we want to test whether we can add the transformers afterwards
    yaml_model.input_transformers = (input_transformer, 0)
    yaml_model.output_transformers = (output_transformer, 0)
    assert yaml_model.input_transformers == [input_transformer]
    assert yaml_model.output_transformers == [output_transformer]


def test_model_from_objects(
    california_model_info: Dict[str, str],
    california_model_kwargs: Dict[str, Union[List, Dict, str]],
    california_variables: Tuple[dict, dict],
    california_transformers: Tuple[list, list],
    cal_model: PyTorchModel,
):
    input_variables, output_variables = california_variables
    input_transformer, output_transformer = california_transformers

    assert cal_model._feature_order == california_model_info["model_in_list"]
    assert cal_model._output_order == california_model_info["model_out_list"]
    assert isinstance(cal_model, PyTorchModel)
    assert cal_model.input_variables == input_variables
    assert cal_model.output_variables == output_variables
    assert cal_model.features == california_model_kwargs["feature_order"]
    assert cal_model.outputs == california_model_kwargs["output_order"]
    assert cal_model.input_transformers == [input_transformer]
    assert cal_model.output_transformers == [output_transformer]


def test_california_housing_model_variable(
    california_test_x_dict: Dict[str, torch.Tensor],
    california_model_kwargs: Dict[str, Union[List, Dict, str]],
):
    args = deepcopy(california_model_kwargs)
    args["output_format"] = {"type": "variable"}
    cal_model = PyTorchModel(**args)

    input_variables_dict = deepcopy(cal_model.input_variables)
    for key, var in input_variables_dict.items():
        var.value = california_test_x_dict[key].item()

    results = cal_model.evaluate(input_variables_dict)

    assert isinstance(results["MedHouseVal"], ScalarOutputVariable)
    assert results["MedHouseVal"].value == pytest.approx(4.063651)
    assert_variables_updated(
        california_test_x_dict["HouseAge"].item(),
        4.063651,
        cal_model,
        "HouseAge",
        "MedHouseVal",
    )


def test_california_housing_model_tensor(
    california_test_x_dict: Dict[str, torch.Tensor], cal_model: PyTorchModel
):
    results = cal_model.evaluate(california_test_x_dict)

    assert torch.isclose(
        results["MedHouseVal"], torch.tensor(4.063651, dtype=torch.double)
    )
    assert isinstance(results["MedHouseVal"], torch.Tensor)
    assert_variables_updated(
        california_test_x_dict["HouseAge"].item(),
        4.063651,
        cal_model,
        "HouseAge",
        "MedHouseVal",
    )


def test_california_housing_model_multi_tensor(
    california_test_x, cal_model: PyTorchModel
):
    test_dict = {
        key: california_test_x[:, idx] for idx, key in enumerate(cal_model.features)
    }
    results = cal_model.evaluate(test_dict)
    # in this case we don't expect the input/output variables to be updated,
    # because we don't know which value to update them with so we only check
    # for the resulting values
    assert all(
        torch.isclose(
            results["MedHouseVal"],
            torch.tensor([4.063651, 2.7774928, 2.792812], dtype=torch.double),
        )
    )


def test_california_housing_model_float(
    california_test_x_dict: Dict[str, torch.Tensor],
    california_model_kwargs: Dict[str, Union[List, Dict, str]],
):
    args = deepcopy(california_model_kwargs)
    args["output_format"] = {"type": "raw"}
    cal_model = PyTorchModel(**args)

    float_dict = {key: value.item() for key, value in california_test_x_dict.items()}

    results = cal_model.evaluate(float_dict)

    assert results["MedHouseVal"] == pytest.approx(4.063651)
    assert isinstance(results["MedHouseVal"], float)
    assert_variables_updated(
        california_test_x_dict["HouseAge"].item(),
        4.063651,
        cal_model,
        "HouseAge",
        "MedHouseVal",
    )


def test_california_housing_model_shuffled_input(
    california_test_x_dict: Dict[str, torch.Tensor], cal_model: PyTorchModel
):
    shuffled_input = deepcopy(california_test_x_dict)
    l = list(shuffled_input.items())
    random.shuffle(l)
    shuffled_input = dict(l)

    results = cal_model.evaluate(shuffled_input)

    assert torch.isclose(
        results["MedHouseVal"], torch.tensor(4.063651, dtype=torch.double)
    )
    assert isinstance(results["MedHouseVal"], torch.Tensor)
    assert_variables_updated(
        california_test_x_dict["HouseAge"].item(),
        4.063651,
        cal_model,
        "HouseAge",
        "MedHouseVal",
    )


@pytest.mark.parametrize(
    "test_idx,expected",
    [
        (0, torch.tensor(4.063651, dtype=torch.double)),
        (1, torch.tensor(2.7774928, dtype=torch.double)),
        (2, torch.tensor(2.792812, dtype=torch.double)),
    ],
)
def test_california_housing_model_execution_diff_values(
    test_idx: int,
    expected: torch.Tensor,
    california_test_x: torch.Tensor,
    cal_model: PyTorchModel,
):
    test_input = {
        key: california_test_x[test_idx][idx]
        for idx, key in enumerate(cal_model.features)
    }

    results = cal_model.evaluate(test_input)

    assert torch.isclose(results["MedHouseVal"], expected)
    assert_variables_updated(
        test_input["HouseAge"].item(),
        expected.item(),
        cal_model,
        "HouseAge",
        "MedHouseVal",
    )


def test_california_housing_model_execution_no_transformation(
    california_test_x_dict: Dict[str, torch.Tensor],
    california_model_kwargs: Dict[str, Union[List, Dict, str]],
):
    # if we don't pass in an output transformer, we expect to get the untransformed
    # result back
    new_kwargs = deepcopy(california_model_kwargs)
    new_kwargs["output_transformers"] = []
    cal_model = PyTorchModel(**new_kwargs)

    results = cal_model.evaluate(california_test_x_dict)

    assert torch.isclose(
        results["MedHouseVal"], torch.tensor(1.8523695, dtype=torch.double)
    )
    assert_variables_updated(
        california_test_x_dict["HouseAge"].item(),
        1.8523695,
        cal_model,
        "HouseAge",
        "MedHouseVal",
    )


def test_california_housing_model_execution_missing_input(
    caplog, california_test_x_dict: Dict[str, torch.Tensor], cal_model: PyTorchModel
):
    missing_dict = deepcopy(california_test_x_dict)
    del missing_dict["Longitude"]

    with caplog.at_level(logging.INFO):
        results = cal_model.evaluate(missing_dict)
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "INFO"
        assert (
            caplog.records[0].message
            == "'Longitude' missing from input_dict, using default value"
        )


def test_differentiability(
    california_test_x_dict: Dict[str, torch.Tensor], cal_model: PyTorchModel
):
    differentiable_dict = deepcopy(california_test_x_dict)
    for value in differentiable_dict.values():
        value.requires_grad = True

    results = cal_model.evaluate(differentiable_dict)

    # if we maintain differentiability, we should be able to call .backward()
    # on a model output without it causing an error
    for key, value in results.items():
        try:
            value.backward()
            assert value.requires_grad
        except AttributeError as exc:
            # if the attribute error is raised because we're, returning a float,
            # the test should fail
            assert False, str(exc)

    # we also want to make sure that the input_variable and output_variable
    # values are still treated as floats
    assert isinstance(results["MedHouseVal"], torch.Tensor)
    assert torch.isclose(
        results["MedHouseVal"], torch.tensor(4.063651, dtype=torch.double)
    )
    assert_variables_updated(
        california_test_x_dict["HouseAge"].item(),
        4.063651,
        cal_model,
        "HouseAge",
        "MedHouseVal",
    )
