import torch

from lume_model.calibrations.torch_model.decoupled_linear import (
    InputOffset,
    InputScale,
    DecoupledLinearInput,
    OutputOffset,
    OutputScale,
    DecoupledLinearOutput,
    DecoupledLinear,
)


class TestInputOffset:
    def test_init(self, linear_model):
        _ = InputOffset(model=linear_model)

    def test_forward(self, linear_model):
        x_offset = torch.rand(1)
        m = InputOffset(model=linear_model, x_offset_initial=x_offset)
        x = torch.rand(1)

        assert torch.isclose(m(x), linear_model(x + x_offset))


class TestInputScale:
    def test_init(self, linear_model):
        _ = InputScale(model=linear_model)

    def test_forward(self, linear_model):
        x_scale = torch.ones(1) + torch.rand(1)
        m = InputScale(model=linear_model, x_scale_initial=x_scale)
        x = torch.rand(1)

        assert torch.isclose(m(x), linear_model(x_scale * x))


class TestDecoupledLinearInput:
    def test_init(self, linear_model):
        _ = DecoupledLinearInput(model=linear_model)

    def test_forward(self, linear_model):
        x_offset, x_scale = torch.rand(1), torch.ones(1) + torch.rand(1)
        m = DecoupledLinearInput(
            model=linear_model, x_offset_initial=x_offset, x_scale_initial=x_scale
        )
        x = torch.rand(1)

        assert torch.isclose(m(x), linear_model(x_scale * (x + x_offset)))


class TestOutputOffset:
    def test_init(self, linear_model):
        _ = OutputOffset(model=linear_model)

    def test_forward(self, linear_model):
        y_offset = torch.rand(1)
        m = OutputOffset(model=linear_model, y_offset_initial=y_offset)
        x = torch.rand(1)

        assert torch.isclose(m(x), linear_model(x) + y_offset)


class TestOutputScale:
    def test_init(self, linear_model):
        _ = OutputScale(model=linear_model)

    def test_forward(self, linear_model):
        y_scale = torch.ones(1) + torch.rand(1)
        m = OutputScale(model=linear_model, y_scale_initial=y_scale)
        x = torch.rand(1)

        assert torch.isclose(m(x), y_scale * linear_model(x))


class TestDecoupledLinearOutput:
    def test_init(self, linear_model):
        _ = DecoupledLinearOutput(model=linear_model)

    def test_forward(self, linear_model):
        y_offset, y_scale = torch.rand(1), torch.ones(1) + torch.rand(1)
        m = DecoupledLinearOutput(
            model=linear_model, y_offset_initial=y_offset, y_scale_initial=y_scale
        )
        x = torch.rand(1)

        assert torch.isclose(m(x), y_scale * (linear_model(x) + y_offset))


class TestDecoupledLinear:
    def test_init(self, linear_model):
        _ = DecoupledLinear(model=linear_model)

    def test_forward(self, one_dim_decoupled_linear_module):
        m = one_dim_decoupled_linear_module
        model = one_dim_decoupled_linear_module.model
        x = torch.rand(1)

        assert torch.isclose(
            m(x), m.y_scale * (model(m.x_scale * (x + m.x_offset)) + m.y_offset)
        )
