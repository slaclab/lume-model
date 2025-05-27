import torch

from lume_model.calibrations.torch_model.decoupled_linear import DecoupledLinear
from lume_model.calibrations.torch_model.utils import (
    extract_transformers,
    get_decoupled_linear_parameters,
)


def test_extract_transformers(one_dim_decoupled_linear_module):
    m = one_dim_decoupled_linear_module
    model = one_dim_decoupled_linear_module.model
    input_transformer, output_transformer = extract_transformers(m)
    x = torch.rand(1).unsqueeze(0)

    def transform(_x):
        _x = input_transformer.transform(_x)
        _x = model(_x)
        return output_transformer.untransform(_x)

    assert torch.isclose(transform(x), m(x))


def test_get_decoupled_linear_parameters(one_dim_decoupled_linear_module):
    m_1 = one_dim_decoupled_linear_module
    model = one_dim_decoupled_linear_module.model
    input_transformer, output_transformer = extract_transformers(m_1)
    kwargs = get_decoupled_linear_parameters(input_transformer, output_transformer)
    m_2 = DecoupledLinear(model=model, **kwargs)
    x = torch.rand(1)

    assert torch.isclose(m_1(x), m_2(x))
