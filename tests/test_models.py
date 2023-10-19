import pytest

try:
    from lume_model.torch import TorchModel, TorchModule
    from lume_model.keras import KerasModel
    from lume_model.models import model_from_yaml
except ImportError:
    pass


@pytest.mark.parametrize("filename,expected", [
    ("test_files/california_regression/torch_model.yml", TorchModel),
    ("test_files/california_regression/torch_module.yml", TorchModule),
    ("test_files/iris_classification/keras_model.yml", KerasModel),
])
def test_model_from_yaml(rootdir, filename, expected):
    model = model_from_yaml(f"{rootdir}/{filename}")
    assert isinstance(model, expected)
