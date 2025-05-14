import pytest
import random

try:
    import torch
    from botorch.models import SingleTaskGP, MultiTaskGP

    torch.manual_seed(42)
except ImportError:
    pass

from lume_model.models.gp_model import GPModel

random.seed(42)


class TestGPModel:
    def test_validates_model_type(self, single_task_gp_model_kwargs):
        with pytest.raises(OSError):
            GPModel(
                model="invalid_model_type",
                input_variables=single_task_gp_model_kwargs["input_variables"],
                output_variables=single_task_gp_model_kwargs["output_variables"],
            )

    def test_raises_error_for_missing_transformer_file(
        self, single_task_gp_model_kwargs
    ):
        with pytest.raises(OSError):
            GPModel(
                model=SingleTaskGP(
                    torch.rand(10, 1, dtype=torch.double),
                    torch.rand(10, 1, dtype=torch.double),
                ),
                input_transformers=["non_existent_transformer.pt"],
                output_transformers=["non_existent_transformer.pt"],
                input_variables=single_task_gp_model_kwargs["input_variables"],
                output_variables=single_task_gp_model_kwargs["output_variables"],
            )

    def test_single_task_gp(self, single_task_gp_model_kwargs):
        gp_model = GPModel(
            model=single_task_gp_model_kwargs["model"],
            input_variables=single_task_gp_model_kwargs["input_variables"],
            output_variables=single_task_gp_model_kwargs["output_variables"],
            input_transformers=single_task_gp_model_kwargs["input_transformers"],
            output_transformers=single_task_gp_model_kwargs["output_transformers"],
        )
        assert isinstance(gp_model.model, SingleTaskGP)

        original_model = single_task_gp_model_kwargs["model"]
        input_transformer = single_task_gp_model_kwargs["input_transformers"][0]
        output_transformer = single_task_gp_model_kwargs["output_transformers"][0]

        test_x = torch.rand(10, 1).to(torch.double)
        # Predict with original model
        test_x_tf = input_transformer.transform(test_x)
        original_pred = original_model.posterior(test_x_tf)
        # Output transformer is a ReversibleInputTransform type
        original_mean = output_transformer.untransform(original_pred.mean)
        original_variance = (
            abs(output_transformer.coefficient) ** 2 * original_pred.variance
        )
        # Predict with GPModel
        lume_pred = gp_model.evaluate({"input": test_x})

        assert gp_model.get_input_size() == 1
        assert gp_model.get_output_size() == 2
        assert len(lume_pred) == 2
        assert torch.allclose(original_mean[:, 0], lume_pred["output1"].mean)
        assert torch.allclose(original_variance[:, 0], lume_pred["output1"].variance)
        assert torch.allclose(original_mean[:, 1], lume_pred["output2"].mean)
        assert torch.allclose(original_variance[:, 1], lume_pred["output2"].variance)

        # Test batched evaluation
        batch_test_x = test_x.reshape(-1, 1).repeat(2, 1, 1)
        # Predict with original model
        batch_test_x_tf = input_transformer.transform(batch_test_x)
        original_pred = original_model.posterior(batch_test_x_tf)
        # Output transformer is a ReversibleInputTransform type
        original_mean = output_transformer.untransform(original_pred.mean)
        original_variance = (
            abs(output_transformer.coefficient) ** 2 * original_pred.variance
        )
        # Predict with GPModel
        lume_pred = gp_model.evaluate({"input": batch_test_x})

        assert lume_pred["output1"].mean.shape == (2, 10) and lume_pred[
            "output1"
        ].variance.shape == (2, 10)
        assert lume_pred["output2"].mean.shape == (2, 10) and lume_pred[
            "output2"
        ].variance.shape == (2, 10)
        assert original_pred.mean.shape == (
            2,
            10,
            2,
        ) and original_pred.variance.shape == (2, 10, 2)
        assert torch.allclose(original_mean[:, :, 0], lume_pred["output1"].mean)
        assert torch.allclose(original_variance[:, :, 0], lume_pred["output1"].variance)
        assert torch.allclose(original_mean[:, :, 1], lume_pred["output2"].mean)
        assert torch.allclose(original_variance[:, :, 1], lume_pred["output2"].variance)

    def test_multi_task_gp(self, multi_task_gp_model_kwargs):
        gp_model = GPModel(
            model=multi_task_gp_model_kwargs["model"],
            input_variables=multi_task_gp_model_kwargs["input_variables"],
            output_variables=multi_task_gp_model_kwargs["output_variables"],
            input_transformers=multi_task_gp_model_kwargs["input_transformers"],
            output_transformers=multi_task_gp_model_kwargs["output_transformers"],
        )
        assert isinstance(gp_model.model, MultiTaskGP)

        original_model = multi_task_gp_model_kwargs["model"]
        input_transformer = multi_task_gp_model_kwargs["input_transformers"][0]
        output_transformer = multi_task_gp_model_kwargs["output_transformers"][0]

        # Make sure to use double precision for all GP tests
        test_x = torch.linspace(0, 1, 200).reshape(-1, 1).to(torch.double)
        # Predict with original model
        test_x_tf = input_transformer.transform(test_x)
        original_pred = original_model.posterior(test_x_tf)
        m = original_pred.mean
        v = original_pred.variance
        # Output transformer is a OutcomeTransform type
        original_mean = output_transformer.stdvs.squeeze(
            0
        ) * m + output_transformer.means.squeeze(0)
        original_variance = abs(output_transformer.stdvs.squeeze(0)) ** 2 * v
        # Predict with GPModel
        lume_pred = gp_model.evaluate({"input": test_x})

        assert gp_model.get_input_size() == 1
        assert gp_model.get_output_size() == 2
        assert len(lume_pred) == 2
        assert lume_pred["output1"].mean.shape == (200,) and lume_pred[
            "output1"
        ].variance.shape == (200,)
        assert lume_pred["output2"].mean.shape == (200,) and lume_pred[
            "output2"
        ].variance.shape == (200,)
        assert original_pred.mean.shape == (
            200,
            2,
        ) and original_pred.variance.shape == (200, 2)
        assert torch.allclose(original_mean[:, 0], lume_pred["output1"].mean)
        assert torch.allclose(original_variance[:, 0], lume_pred["output1"].variance)
        assert torch.allclose(original_mean[:, 1], lume_pred["output2"].mean)
        assert torch.allclose(original_variance[:, 1], lume_pred["output2"].variance)

        # Test batched evaluation
        batch_test_x = test_x.repeat(2, 1, 1)
        # Predict with original model
        batch_test_x_tf = input_transformer.transform(batch_test_x)
        original_pred = original_model.posterior(batch_test_x_tf)
        m = original_pred.mean
        v = original_pred.variance
        # Output transformer is a OutcomeTransform type
        original_mean = output_transformer.stdvs.squeeze(
            0
        ) * m + output_transformer.means.squeeze(0)
        original_variance = abs(output_transformer.stdvs.squeeze(0)) ** 2 * v
        # Predict with GPModel
        lume_pred = gp_model.evaluate({"input": batch_test_x})

        assert lume_pred["output1"].mean.shape == (2, 200) and lume_pred[
            "output1"
        ].variance.shape == (2, 200)
        assert lume_pred["output2"].mean.shape == (2, 200) and lume_pred[
            "output2"
        ].variance.shape == (2, 200)
        assert original_pred.mean.shape == (
            2,
            200,
            2,
        ) and original_pred.variance.shape == (2, 200, 2)
        assert torch.allclose(original_mean[:, :, 0], lume_pred["output1"].mean)
        assert torch.allclose(original_variance[:, :, 0], lume_pred["output1"].variance)
        assert torch.allclose(original_mean[:, :, 1], lume_pred["output2"].mean)
        assert torch.allclose(original_variance[:, :, 1], lume_pred["output2"].variance)
