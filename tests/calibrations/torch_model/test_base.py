import os
import importlib

import torch
from gpytorch.priors import NormalPrior
from gpytorch.constraints import Interval

import lume_model.calibrations.torch_model.base as base
from lume_model.calibrations.torch_model.base import ParameterModule


def assert_parameter_init(m: ParameterModule, name: str):
    for attr in [
        f"raw_{name}",
        name,
        f"_{name}_default",
        f"_{name}_initial",
        f"{name}_mask",
    ]:
        assert hasattr(m, attr)
    assert name in m.calibration_parameter_names
    assert (
        len(m.calibration_parameter_names)
        == len(m.raw_calibration_parameters)
        == len(m.calibration_parameters)
    )
    raw_param = getattr(m, f"raw_{name}")
    assert isinstance(raw_param, torch.nn.parameter.Parameter)
    assert raw_param.requires_grad
    param_is_default = all(
        torch.isclose(getattr(m, name), getattr(m, f"_{name}_default")).flatten()
    )
    if param_is_default:
        assert all(
            torch.isclose(raw_param.data, torch.zeros(raw_param.shape)).flatten()
        )
    assert (f"raw_{name}", raw_param) in m.named_parameters()


class TestParameterModule:
    def test_init(self, linear_model):
        m = ParameterModule(model=linear_model)

        assert not m.calibration_parameter_names

    def test_init_single_parameter(self, linear_model, parameter_name):
        m = ParameterModule(model=linear_model, parameter_names=[parameter_name])

        assert_parameter_init(m, parameter_name)

    def test_init_multiple_parameters(self, linear_model, parameter_names):
        m = ParameterModule(model=linear_model, parameter_names=parameter_names)

        for name in parameter_names:
            assert_parameter_init(m, name)

    def test_parameter_default(self, linear_model, parameter_name):
        m = ParameterModule(
            model=linear_model,
            parameter_names=[parameter_name],
            **{f"{parameter_name}_default": 1.0},
        )
        param = getattr(m, parameter_name)
        raw_param = getattr(m, f"raw_{parameter_name}")

        assert torch.isclose(param, torch.zeros(param.shape))
        assert torch.isclose(raw_param.data, -torch.ones(raw_param.shape))

    def test_parameter_initial(self, linear_model, parameter_name):
        m = ParameterModule(
            model=linear_model,
            parameter_names=[parameter_name],
            **{f"{parameter_name}_initial": 1.0},
        )
        param = getattr(m, parameter_name)
        raw_param = getattr(m, f"raw_{parameter_name}")

        assert torch.isclose(param, torch.ones(param.shape))
        assert torch.isclose(raw_param.data, torch.ones(raw_param.shape))

    def test_parameter_prior(self, linear_model, parameter_name):
        m = ParameterModule(
            model=linear_model,
            parameter_names=[parameter_name],
            **{
                f"{parameter_name}_prior": NormalPrior(
                    loc=torch.zeros((1, 1)), scale=torch.ones((1, 1))
                )
            },
        )

        assert hasattr(m, f"{parameter_name}_prior")
        prior_module = getattr(m, f"{parameter_name}_prior")
        assert issubclass(prior_module.__class__, torch.nn.Module)
        assert prior_module in m.modules()
        assert f"{parameter_name}_prior" in list(zip(*m.named_priors()))[0]
        assert prior_module in list(zip(*m.named_priors()))[2]

    def test_parameter_constraint(self, linear_model, parameter_name):
        kwargs = {
            f"{parameter_name}_default": 1.0,
            f"{parameter_name}_initial": 1.0,
            f"{parameter_name}_constraint": Interval(lower_bound=-1.5, upper_bound=1.5),
        }
        m = ParameterModule(
            model=linear_model,
            parameter_names=[parameter_name],
            **kwargs,
        )
        param = getattr(m, parameter_name)
        raw_param = getattr(m, f"raw_{parameter_name}")

        assert hasattr(m, f"raw_{parameter_name}_constraint")
        c = getattr(m, f"raw_{parameter_name}_constraint")
        assert (f"raw_{parameter_name}_constraint", c) in m.named_constraints()
        assert c.lower_bound < param < c.upper_bound
        assert torch.isclose(raw_param.data, torch.zeros(raw_param.shape))
        assert torch.isclose(param, torch.ones(param.shape))

    def test_ndim_parameter(self, linear_model, parameter_name, ndim_size):
        parameter_prior = NormalPrior(
            loc=torch.zeros(ndim_size), scale=torch.ones(ndim_size)
        )
        kwargs = {
            f"{parameter_name}_size": ndim_size,
            f"{parameter_name}_default": torch.ones(ndim_size),
            f"{parameter_name}_initial": torch.ones(ndim_size),
            f"{parameter_name}_prior": parameter_prior,
            f"{parameter_name}_constraint": Interval(lower_bound=-1.5, upper_bound=1.5),
        }
        m = ParameterModule(
            model=linear_model,
            parameter_names=[parameter_name],
            **kwargs,
        )
        param = getattr(m, parameter_name)
        raw_param = getattr(m, f"raw_{parameter_name}")

        assert param.shape == ndim_size
        assert raw_param.shape == ndim_size

    def test_parameter_lists(self, linear_model, parameter_name, ndim_size):
        parameter_prior = NormalPrior(
            loc=torch.zeros(ndim_size), scale=torch.ones(ndim_size)
        )
        kwargs = {
            f"{parameter_name}_size": ndim_size,
            f"{parameter_name}_default": torch.ones(ndim_size).tolist(),
            f"{parameter_name}_initial": torch.ones(ndim_size).tolist(),
            f"{parameter_name}_prior": parameter_prior,
            f"{parameter_name}_constraint": Interval(lower_bound=-1.5, upper_bound=1.5),
        }
        m = ParameterModule(
            model=linear_model,
            parameter_names=[parameter_name],
            **kwargs,
        )
        param = getattr(m, parameter_name)
        raw_param = getattr(m, f"raw_{parameter_name}")

        assert param.shape == ndim_size
        assert raw_param.shape == ndim_size

    def test_parameter_mask(self, extensive_parameter_module):
        parameter_name = extensive_parameter_module.calibration_parameter_names[0]
        param = extensive_parameter_module.calibration_parameters[0]
        mask = getattr(extensive_parameter_module, f"{parameter_name}_mask")

        assert all(
            torch.isclose(
                param[mask], torch.ones(torch.count_nonzero(mask), dtype=param.dtype)
            )
        )
        assert all(
            torch.isclose(
                param[~mask], torch.zeros(torch.count_nonzero(~mask), dtype=param.dtype)
            )
        )

    def test_add_parameter_name_to_kwargs(self, linear_model, parameter_names):
        kwargs, kwargs_n = {}, {}
        m = ParameterModule(model=linear_model)
        m._add_parameter_name_to_kwargs(parameter_names[0], kwargs)
        for name in parameter_names:
            m._add_parameter_name_to_kwargs(name, kwargs_n)

        assert parameter_names[0] in kwargs.get("parameter_names")
        assert all(name in kwargs_n.get("parameter_names") for name in parameter_names)

    def test_torch_save_and_load(self, extensive_parameter_module, tmp_path):
        f = os.path.join(tmp_path, "test_module.pt")
        torch.save(extensive_parameter_module, f)
        importlib.reload(base)  # refresh ParameterModule class definition
        m = torch.load(f, weights_only=False)
        os.remove(f)

        assert str(m.state_dict()) == str(extensive_parameter_module.state_dict())
        for name in extensive_parameter_module.calibration_parameter_names:
            assert_parameter_init(m, name)
