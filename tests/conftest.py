import os
import json
from typing import Any, Union

import pytest

from lume_model.utils import variables_from_yaml
from lume_model.variables import ScalarVariable

try:
    import torch
    from botorch.models.transforms.input import AffineInputTransform  # noqa: F401
    from lume_model.models import TorchModel, TorchModule
except ModuleNotFoundError:
    pass


@pytest.fixture(scope="session")
def rootdir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def simple_variables() -> dict[str, Union[list[ScalarVariable], list[ScalarVariable]]]:
    input_variables = [
        ScalarVariable(name="input1", default_value=1.0, value_range=(0.0, 5.0)),
        ScalarVariable(name="input2", default_value=2.0, value_range=(1.0, 3.0)),
    ]
    output_variables = [ScalarVariable(name="output1"), ScalarVariable(name="output2")]
    return {"input_variables": input_variables, "output_variables": output_variables}


@pytest.fixture(scope="module")
def california_model_info(rootdir) -> dict[str, str]:
    try:
        with open(
            f"{rootdir}/test_files/california_regression/model_info.json", "r"
        ) as f:
            model_info = json.load(f)
        return model_info
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def california_variables(rootdir) -> tuple[list[ScalarVariable], list[ScalarVariable]]:
    try:
        file = f"{rootdir}/test_files/california_regression/variables.yml"
        input_variables, output_variables = variables_from_yaml(file)
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
    rootdir,
    california_model_info,
    california_variables,
    california_transformers,
) -> dict[str, Any]:
    _ = pytest.importorskip("botorch")

    input_variables, output_variables = california_variables
    input_transformer, output_transformer = california_transformers
    model_kwargs = {
        "model": torch.load(
            f"{rootdir}/test_files/california_regression/model.pt", weights_only=False
        ),
        "input_variables": input_variables,
        "output_variables": output_variables,
        "input_transformers": [input_transformer],
        "output_transformers": [output_transformer],
        "output_format": "tensor",
    }
    return model_kwargs


@pytest.fixture(scope="module")
def california_test_input_tensor(rootdir: str):
    torch = pytest.importorskip("torch")

    try:
        test_input_tensor = torch.load(
            f"{rootdir}/test_files/california_regression/test_input_tensor.pt",
            weights_only=False,
        )
    except FileNotFoundError as e:
        pytest.skip(str(e))
    return test_input_tensor


@pytest.fixture(scope="module")
def california_test_input_dict(
    california_test_input_tensor, california_model_info
) -> dict:
    pytest.importorskip("botorch")

    test_input_dict = {
        key: california_test_input_tensor[0, idx]
        for idx, key in enumerate(california_model_info["model_in_list"])
    }
    return test_input_dict


@pytest.fixture(scope="module")
def california_model(california_model_kwargs):
    _ = pytest.importorskip("botorch")

    return TorchModel(**california_model_kwargs)


@pytest.fixture(scope="module")
def california_module(california_model):
    _ = pytest.importorskip("botorch")

    return TorchModule(model=california_model)
