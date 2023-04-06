import json
import os
from typing import Dict, List, TextIO, Tuple, Union

import pytest

from lume_model.utils import variables_from_yaml
from lume_model.variables import InputVariable, OutputVariable

try:
    import torch
    from botorch.models.transforms.input import AffineInputTransform

    from lume_model.torch import PyTorchModel
except ModuleNotFoundError:
    pass


@pytest.fixture(scope="session")
def rootdir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def config_file(rootdir) -> TextIO:
    try:
        return open(f"{rootdir}/test_files/iris_config.yml", "r")
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def california_model_info(rootdir) -> Dict[str, str]:
    try:
        with open(
            f"{rootdir}/test_files/california_regression/model_info.json", "r"
        ) as f:
            model_info = json.load(f)
        return model_info
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def california_variables(
    rootdir,
) -> Tuple[Dict[str, InputVariable], Dict[str, OutputVariable]]:
    try:
        with open(
            f"{rootdir}/test_files/california_regression/california_variables.yml", "r"
        ) as f:
            input_variables, output_variables = variables_from_yaml(f)
        return input_variables, output_variables
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def california_transformers(rootdir):
    botorch = pytest.importorskip("botorch")
    try:
        with open(
            f"{rootdir}/test_files/california_regression/normalization.json", "r"
        ) as f:
            normalizations = json.load(f)
    except FileNotFoundError as e:
        pytest.skip(str(e))

    input_transformer = botorch.models.transforms.input.AffineInputTransform(
        len(normalizations["x_mean"]),
        coefficient=torch.tensor(normalizations["x_scale"]),
        offset=torch.tensor(normalizations["x_mean"]),
    )
    output_transformer = botorch.models.transforms.input.AffineInputTransform(
        len(normalizations["y_mean"]),
        coefficient=torch.tensor(normalizations["y_scale"]),
        offset=torch.tensor(normalizations["y_mean"]),
    )
    return input_transformer, output_transformer


@pytest.fixture(scope="module")
def california_model_kwargs(
    rootdir, california_model_info, california_variables, california_transformers
) -> Dict[str, Union[List, Dict, str]]:
    botorch = pytest.importorskip("botorch")

    input_variables, output_variables = california_variables
    input_transformer, output_transformer = california_transformers
    model_kwargs = {
        "model_file": f"{rootdir}/test_files/california_regression/california_regression.pt",
        "input_variables": input_variables,
        "output_variables": output_variables,
        "input_transformers": [input_transformer],
        "output_transformers": [output_transformer],
        "feature_order": california_model_info["model_in_list"],
        "output_order": california_model_info["model_out_list"],
        "output_format": {"type": "tensor"},
    }
    return model_kwargs


@pytest.fixture(scope="module")
def california_test_x(rootdir: str):
    torch = pytest.importorskip("torch")

    try:
        test_x = torch.load(f"{rootdir}/test_files/california_regression/X_test_raw.pt")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    # for speed/memory in tests we set requires grad to false and only activate it
    # when testing for differentiability
    test_x.requires_grad = False
    return test_x


@pytest.fixture(scope="module")
def california_test_x_dict(california_test_x, california_model_info):
    pytest.importorskip("botorch")
    test_x_dict = {
        key: california_test_x[0][idx]
        for idx, key in enumerate(california_model_info["model_in_list"])
    }
    return test_x_dict


@pytest.fixture
def cal_model(california_model_kwargs):
    botorch = pytest.importorskip("botorch")
    model = PyTorchModel(**california_model_kwargs)
    return model
