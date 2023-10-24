import os
import random
from typing import Any, Union
from copy import deepcopy

import pytest

try:
    import torch
    from botorch.models.transforms.input import AffineInputTransform
    from lume_model.models import TorchModel
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
#     have been updated correctly with float values (NOT tensors)"""
#     assert isinstance(model.input_variables[model.input_names.index(input_name)].value, float)
#     assert model.input_variables[model.input_names.index(input_name)].value == pytest.approx(input_value)
#     assert isinstance(model.output_variables[model.output_names.index(output_name)].value, float)
#     assert model.output_variables[model.output_names.index(output_name)].value == pytest.approx(output_value)


# def assert_california_model_result(california_test_input_dict: dict, model):
#     assert_variables_updated(
#         input_value=california_test_input_dict["HouseAge"].item(),
#         output_value=4.063651,
#         model=model,
#         input_name="HouseAge",
#         output_name="MedHouseVal",
#     )


def assert_model_equality(m1: TorchModel, m2: TorchModel):
    assert m1.input_variables == m2.input_variables
    assert m1.output_variables == m2.output_variables
    assert str(m1.model.state_dict()) == str(m2.model.state_dict())
    for attr in ["input_transformers", "output_transformers"]:
        m1_transformers, m2_transformers = getattr(m1, attr), getattr(m2, attr)
        assert len(m1_transformers) == len(m2_transformers)
        for i, t in enumerate(m1_transformers):
            if isinstance(t, torch.nn.Module):
                assert isinstance(m2_transformers[i], torch.nn.Module)
                assert str(t.state_dict()) == str(m2_transformers[i].state_dict())
    assert m1.output_format == m2.output_format
    assert m1.device == m2.device
    assert m1.fixed_model == m2.fixed_model


class TestTorchModel:
    def test_model_from_objects(
            self,
            california_model_info: dict[str, str],
            california_model_kwargs: dict[str, Union[list, dict, str]],
            california_variables: tuple[list[InputVariable], list[OutputVariable]],
            california_transformers: tuple[list, list],
            california_model,
    ):
        input_variables, output_variables = california_variables
        input_transformer, output_transformer = california_transformers

        assert isinstance(california_model, TorchModel)
        assert california_model.input_names == california_model_info["model_in_list"]
        assert california_model.output_names == california_model_info["model_out_list"]
        assert california_model.input_variables == input_variables
        assert california_model.output_variables == output_variables
        assert california_model.input_transformers == [input_transformer]
        assert california_model.output_transformers == [output_transformer]

    def test_model_from_yaml(self, rootdir: str, california_model):
        file = os.path.join(
            rootdir, "test_files", "california_regression", "torch_model.yml"
        )
        yaml_model = TorchModel.from_file(file)

        assert_model_equality(yaml_model, california_model)

    def test_model_as_yaml(self, rootdir: str, california_model):
        filename = "test_torch_model"
        file = f"{filename}.yml"
        california_model.dump(file)
        yaml_model = TorchModel(file)
        assert_model_equality(yaml_model, california_model)
        os.remove(file)
        os.remove(f"{filename}_model.pt")
        os.remove(f"{filename}_input_transformers_0.pt")
        os.remove(f"{filename}_output_transformers_0.pt")

    def test_model_evaluate_variable(
            self,
            california_test_input_dict: dict,
            california_model_kwargs: dict[str, Union[list, dict, str]],
    ):
        kwargs = deepcopy(california_model_kwargs)
        kwargs["output_format"] = "variable"
        california_model = TorchModel(**kwargs)
        input_variables = deepcopy(california_model.input_variables)
        for var in input_variables:
            var.value = california_test_input_dict[var.name].item()
        results = california_model.evaluate({var.name: var for var in input_variables})

        assert isinstance(results["MedHouseVal"], ScalarOutputVariable)
        assert results["MedHouseVal"].value == pytest.approx(4.063651)
        # assert_california_model_result(california_test_input_dict, california_model)

    def test_model_evaluate_single_sample(self, california_test_input_dict: dict, california_model):
        results = california_model.evaluate(california_test_input_dict)

        assert isinstance(results["MedHouseVal"], torch.Tensor)
        assert torch.isclose(
            results["MedHouseVal"], torch.tensor(4.063651, dtype=results["MedHouseVal"].dtype)
        )
        # assert_california_model_result(california_test_input_dict, california_model)

    def test_model_evaluate_n_samples(self, california_test_input_tensor, california_model):
        test_dict = {
            key: california_test_input_tensor[:, idx] for idx, key in enumerate(california_model.input_names)
        }
        results = california_model.evaluate(test_dict)
        # in this case we don't expect the input/output variables to be updated, because we don't know which value
        # to update them with so we only check for the resulting values
        target_tensor = torch.tensor([4.063651, 2.7774928, 2.792812], dtype=results["MedHouseVal"].dtype)

        assert all(torch.isclose(results["MedHouseVal"], target_tensor))

    def test_model_evaluate_batch_n_samples(
            self,
            california_test_input_tensor,
            california_model,
    ):
        # model should be able to handle input of shape [n_batch, n_samples, n_dim]
        input_dict = {
            key: california_test_input_tensor[:, idx].unsqueeze(-1).unsqueeze(1).repeat((1, 3, 1))
            for idx, key in enumerate(california_model.input_names)
        }
        results = california_model.evaluate(input_dict)

        # output shape should be [n_batch, n_samples]
        assert tuple(results["MedHouseVal"].shape) == (3, 3)

    def test_model_evaluate_raw(
            self,
            california_test_input_dict: dict,
            california_model_kwargs: dict[str, Union[list, dict, str]],
    ):
        kwargs = deepcopy(california_model_kwargs)
        kwargs["output_format"] = "raw"
        california_model = TorchModel(**kwargs)
        float_dict = {key: value.item() for key, value in california_test_input_dict.items()}
        results = california_model.evaluate(float_dict)

        assert isinstance(results["MedHouseVal"], float)
        assert results["MedHouseVal"] == pytest.approx(4.063651)
        # assert_california_model_result(california_test_input_dict, california_model)

    def test_model_evaluate_shuffled_input(self, california_test_input_dict: dict, california_model):
        shuffled_input = deepcopy(california_test_input_dict)
        item_list = list(shuffled_input.items())
        random.shuffle(item_list)
        shuffled_input = dict(item_list)
        results = california_model.evaluate(shuffled_input)

        assert isinstance(results["MedHouseVal"], torch.Tensor)
        assert torch.isclose(
            results["MedHouseVal"], torch.tensor(4.063651, dtype=results["MedHouseVal"].dtype)
        )
        # assert_california_model_result(california_test_input_dict, california_model)

    @pytest.mark.parametrize("test_idx,expected", [(0, 4.063651), (1, 2.7774928), (2, 2.792812)])
    def test_model_evaluate_different_values(
            self,
            test_idx: int,
            expected: float,
            california_test_input_tensor,
            california_model,
    ):
        input_dict = {
            key: california_test_input_tensor[test_idx][idx] for idx, key in
            enumerate(california_model.input_names)
        }
        results = california_model.evaluate(input_dict)

        assert results["MedHouseVal"].item() == pytest.approx(expected)
        # assert_variables_updated(
        #     input_value=input_dict["HouseAge"].item(),
        #     output_value=expected,
        #     model=california_model,
        #     input_name="HouseAge",
        #     output_name="MedHouseVal",
        # )

    def test_model_evaluate_with_no_output_transformers(
            self,
            california_test_input_dict: dict,
            california_model_kwargs: dict[str, Union[list, dict, str]],
    ):
        kwargs = deepcopy(california_model_kwargs)
        kwargs["output_transformers"] = []
        model = TorchModel(**kwargs)
        results = model.evaluate(california_test_input_dict)

        assert torch.isclose(
            results["MedHouseVal"], torch.tensor(1.8523695, dtype=results["MedHouseVal"].dtype)
        )
        # assert_variables_updated(
        #     input_value=california_test_input_dict["HouseAge"].item(),
        #     output_value=1.8523695,
        #     model=model,
        #     input_name="HouseAge",
        #     output_name="MedHouseVal",
        # )

    def test_differentiability(
            self,
            california_test_input_dict: dict,
            california_model_kwargs: dict[str, Any],
    ):
        kwargs = deepcopy(california_model_kwargs)
        kwargs["model"].train().requires_grad_(True)
        model = TorchModel(**kwargs, fixed_model=False)
        parameters_with_requires_grad = []
        for name, param in model.model.named_parameters():
            if param.requires_grad:
                parameters_with_requires_grad.append(name)
        criterion = torch.nn.MSELoss()

        assert model.model.training
        assert len(parameters_with_requires_grad) == 8
        output_dict = model.evaluate(california_test_input_dict)
        outputs = torch.stack([v for k, v in output_dict.items()])
        loss = criterion(outputs, torch.zeros(outputs.shape, dtype=outputs.dtype))
        loss.backward()

    def test_update_input_variables_to_transformer(self, california_model):
        model = deepcopy(california_model)
        input_variables = model.input_variables

        def get_x_limits(v):
            x_limits = {
                "min": torch.tensor([var.value_range[0] for var in v], dtype=california_model.dtype),
                "max": torch.tensor([var.value_range[1] for var in v], dtype=california_model.dtype),
                "default": torch.tensor([var.default for var in v], dtype=california_model.dtype),
            }
            return x_limits

        x_lim = get_x_limits(input_variables)
        x_lim_nn = {key: model._transform_inputs(x_lim[key]) for key in x_lim.keys()}
        # add new transformer
        d = len(model.input_variables)
        new_transformer = AffineInputTransform(d=d, offset=torch.rand(d), coefficient=1.0 + torch.rand(d))
        model.insert_input_transformer(new_transformer, loc=len(model.input_transformers))
        updated_input_variables = model.update_input_variables_to_transformer(len(model.input_transformers) - 1)
        model.input_variables = updated_input_variables
        # compute new NN limits
        x_lim_updated = get_x_limits(updated_input_variables)
        x_lim_nn_updated = {key: model._transform_inputs(x_lim_updated[key]) for key in x_lim_updated.keys()}

        for key in x_lim_nn.keys():
            assert all(torch.isclose(x_lim_nn[key], x_lim_nn_updated[key], atol=1e-6))
