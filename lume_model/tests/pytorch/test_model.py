import json
from pathlib import Path
from pprint import pprint
from copy import deepcopy

import pytest
import torch
from botorch.models.transforms.input import AffineInputTransform

from lume_model.pytorch import PyTorchModel
from lume_model.utils import variables_from_yaml, model_from_yaml
from lume_model.variables import ScalarInputVariable, ScalarOutputVariable

"""
Things to Test:
---------------
- [x] we can load a PyTorch model from a yaml file
    - [x] returning the model class and keywords
    - [x] returning the model instance
- [x] we can create a PyTorch model from objects
- [x] pytorch model can be run using dictionary of floats or varaibles 
    as input
- [x] pytorch model evaluate() can return either raw or Variable dictionary
- [x] pytorch model can be run with transformers or without
- [x] if we pass in a dictionary that's missing a value, we log 
    an error and use the default value for the input
- [ ] passing different input dictionaries through gives us different
    output dictionaries
- [ ] output transformations are applied in the correct order when we 
    have multiple transformations
"""
tests_dir = str(Path(__file__).parent.parent)
print(tests_dir)

with open(f"{tests_dir}/test_files/california_regression/model_info.json", "r") as f:
    model_info = json.load(f)

with open(
    f"{tests_dir}/test_files/california_regression/california_variables.yml", "r"
) as f:
    input_variables, output_variables = variables_from_yaml(f)

with open(f"{tests_dir}/test_files/california_regression/normalization.json", "r") as f:
    normalizations = json.load(f)

input_transformer = AffineInputTransform(
    len(normalizations["x_mean"]),
    coefficient=torch.Tensor(normalizations["x_scale"]),
    offset=torch.Tensor(normalizations["x_mean"]),
)
output_transformer = AffineInputTransform(
    len(normalizations["y_mean"]),
    coefficient=torch.Tensor(normalizations["y_scale"]),
    offset=torch.Tensor(normalizations["y_mean"]),
)
model_kwargs = {
    "model_file": f"{tests_dir}/test_files/california_regression/california_regression.pt",
    "input_variables": input_variables,
    "output_variables": output_variables,
    "input_transformers": [input_transformer],
    "output_transformers": [output_transformer],
    "feature_order": model_info["model_in_list"],
    "output_order": model_info["model_out_list"],
}
test_x = torch.load(f"{tests_dir}/test_files/california_regression/X_test_raw.pt")
test_x_dict = {
    key: test_x[0][idx].item() for idx, key in enumerate(model_info["model_in_list"])
}


def test_model_from_yaml():
    with open(
        f"{tests_dir}/test_files/california_regression/california_variables.yml", "r"
    ) as f:
        test_model, test_model_kwargs = model_from_yaml(f, load_model=False)

    assert test_model == PyTorchModel
    for key in list(model_kwargs.keys()):
        # we don't define anything about the transformers in the yml file so we
        # don't expect there to be anything in the model_kwargs about them
        if key not in ["input_transformers", "output_transformers"]:
            assert key in list(test_model_kwargs.keys())


def test_model_from_yaml_load_model():
    with open(
        f"{tests_dir}/test_files/california_regression/california_variables.yml", "r"
    ) as f:
        test_model = model_from_yaml(f, load_model=True)

    assert isinstance(test_model, PyTorchModel)
    assert test_model.input_variables == input_variables
    assert test_model.output_variables == output_variables
    assert test_model.features == model_kwargs["feature_order"]
    assert test_model.outputs == model_kwargs["output_order"]
    assert test_model._input_transformers == []
    assert test_model._output_transformers == []


def test_california_housing_model_construction():
    cal_model = PyTorchModel(**model_kwargs)

    assert cal_model._feature_order == model_info["model_in_list"]
    assert cal_model._output_order == model_info["model_out_list"]


def test_california_housing_model_execution_variables():
    cal_model = PyTorchModel(**model_kwargs)

    input_variables_dict = deepcopy(cal_model.input_variables)
    for key, var in input_variables_dict.items():
        var.value = test_x_dict[key]

    results = cal_model.evaluate(input_variables_dict, return_raw=False)

    assert isinstance(results["MedHouseVal"], ScalarOutputVariable)
    assert results["MedHouseVal"].value == pytest.approx(4.063651)

    # we also want to check that the input/output variables have been
    # updated with the right values
    assert cal_model.input_variables["HouseAge"].value == pytest.approx(
        test_x_dict["HouseAge"]
    )
    assert cal_model.output_variables["MedHouseVal"].value == pytest.approx(4.063651)


def test_california_housing_model_execution_raw_values():
    cal_model = PyTorchModel(**model_kwargs)

    results = cal_model.evaluate(test_x_dict, return_raw=True)

    assert results["MedHouseVal"] == pytest.approx(4.063651)
    assert isinstance(results["MedHouseVal"], float)
    assert cal_model.input_variables["HouseAge"].value == pytest.approx(
        test_x_dict["HouseAge"]
    )
    # make sure that the output_variables have been updated as well
    assert cal_model.output_variables["MedHouseVal"].value == pytest.approx(
        pytest.approx(4.063651)
    )


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            {
                key: test_x[0][idx].item()
                for idx, key in enumerate(model_info["model_in_list"])
            },
            4.063651,
        ),
        (
            {
                key: test_x[1][idx].item()
                for idx, key in enumerate(model_info["model_in_list"])
            },
            2.7774928,
        ),
        (
            {
                key: test_x[2][idx].item()
                for idx, key in enumerate(model_info["model_in_list"])
            },
            2.792812,
        ),
    ],
)
def test_california_housing_model_execution_diff_values(test_input, expected):
    cal_model = PyTorchModel(**model_kwargs)

    results = cal_model.evaluate(test_input, return_raw=True)

    assert results["MedHouseVal"] == pytest.approx(expected)
    assert isinstance(results["MedHouseVal"], float)
    assert cal_model.input_variables["HouseAge"].value == pytest.approx(
        test_input["HouseAge"]
    )
    # make sure that the output_variables have been updated as well
    assert cal_model.output_variables["MedHouseVal"].value == pytest.approx(
        pytest.approx(expected)
    )


def test_california_housing_model_execution_no_transformation():
    # if we don't pass in an output transformer, we expect to get the untransformed
    # result back
    new_kwargs = deepcopy(model_kwargs)
    new_kwargs["output_transformers"] = []
    cal_model = PyTorchModel(**new_kwargs)

    results = cal_model.evaluate(test_x_dict, return_raw=True)

    assert results["MedHouseVal"] == pytest.approx(1.8523695)
    assert isinstance(results["MedHouseVal"], float)
    assert cal_model.input_variables["HouseAge"].value == pytest.approx(
        test_x_dict["HouseAge"]
    )
    assert cal_model.output_variables["MedHouseVal"].value == pytest.approx(
        pytest.approx(1.8523695)
    )


def test_california_housing_model_execution_missing_input(caplog):
    cal_model = PyTorchModel(**model_kwargs)

    missing_dict = deepcopy(test_x_dict)
    del missing_dict["Longitude"]

    results = cal_model.evaluate(missing_dict, return_raw=True)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert (
        caplog.records[0].message
        == "'Longitude' missing from input_dict, using default value"
    )
