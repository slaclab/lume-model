import os
import pytest
import random
from copy import deepcopy

try:
    import torch
    from botorch.models import SingleTaskGP, MultiTaskGP, ModelListGP

    torch.manual_seed(42)
except ImportError:
    pass

from lume_model.models.gp_model import GPModel
from lume_model.variables import ScalarVariable, DistributionVariable

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

        # Test with observation noise set to True
        original_pred = original_model.posterior(test_x_tf, observation_noise=True)
        # Output transformer is a ReversibleInputTransform type
        original_mean = output_transformer.untransform(original_pred.mean)
        original_variance = (
            abs(output_transformer.coefficient) ** 2 * original_pred.variance
        )
        # Predict with GPModel
        lume_pred = gp_model.evaluate({"input": test_x}, observation_noise=True)

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

        # Dump and load the model
        gp_model.dump("test.yml")
        loaded_model = GPModel("test.yml")
        lume_pred_loaded = loaded_model.evaluate({"input": batch_test_x})
        assert torch.allclose(
            lume_pred["output1"].mean, lume_pred_loaded["output1"].mean
        )
        assert torch.allclose(
            lume_pred["output1"].variance, lume_pred_loaded["output1"].variance
        )
        assert torch.allclose(
            lume_pred["output2"].mean, lume_pred_loaded["output2"].mean
        )
        assert torch.allclose(
            lume_pred["output2"].variance, lume_pred_loaded["output2"].variance
        )
        os.remove("test.yml")
        os.remove("test_model.pt")
        os.remove("test_input_transformers_0.pt")
        os.remove("test_output_transformers_0.pt")

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

        # Dump and load the model
        gp_model.dump("test.yml")
        loaded_model = GPModel("test.yml")
        lume_pred_loaded = loaded_model.evaluate({"input": batch_test_x})
        assert torch.allclose(
            lume_pred["output1"].mean, lume_pred_loaded["output1"].mean
        )
        assert torch.allclose(
            lume_pred["output1"].variance, lume_pred_loaded["output1"].variance
        )
        assert torch.allclose(
            lume_pred["output2"].mean, lume_pred_loaded["output2"].mean
        )
        assert torch.allclose(
            lume_pred["output2"].variance, lume_pred_loaded["output2"].variance
        )
        os.remove("test.yml")
        os.remove("test_model.pt")
        os.remove("test_input_transformers_0.pt")
        os.remove("test_output_transformers_0.pt")

    def test_model_list_gp(self, create_single_task_gp, create_multi_task_gp):
        # based on botorch/test/models/test_model_list_gp_regression.py

        # SingleTaskGP
        model1, test_x = create_single_task_gp
        model_list_gp = ModelListGP(model1)
        posterior = model_list_gp.posterior(test_x)

        input_variables = [ScalarVariable(name="input")]
        output_variables = [DistributionVariable(name="output")]
        gp_lume_model = GPModel(
            model=model_list_gp,
            input_variables=input_variables,
            output_variables=output_variables,
        )
        lume_pred = gp_lume_model.evaluate({"input": test_x})
        assert gp_lume_model.get_input_size() == 1
        assert gp_lume_model.get_output_size() == 1
        assert len(lume_pred) == 1
        assert torch.allclose(posterior.mean, lume_pred["output"].mean.unsqueeze(-1))
        assert torch.allclose(
            posterior.variance, lume_pred["output"].variance.unsqueeze(-1)
        )

        # MultiTaskGP
        model, model2, train_x_raw = create_multi_task_gp

        # Wrap a single single-output MTGP.
        model_list_gp = ModelListGP(model)
        with torch.no_grad():
            posterior = model_list_gp.posterior(train_x_raw)
        input_variables = [ScalarVariable(name="input1")]
        output_variables = [DistributionVariable(name="output1")]
        gp_lume_model = GPModel(
            model=model_list_gp,
            input_variables=input_variables,
            output_variables=output_variables,
        )
        input_dict = {"input1": train_x_raw}
        output_dict = gp_lume_model.evaluate(input_dict)
        assert gp_lume_model.get_input_size() == 1
        assert gp_lume_model.get_output_size() == 1
        assert len(output_dict) == 1
        assert torch.allclose(output_dict["output1"].mean, posterior.mean.squeeze(-1))
        assert torch.allclose(
            output_dict["output1"].variance, posterior.variance.squeeze(-1)
        )

        # Wrap two single-output MTGPs.
        model_list_gp = ModelListGP(model, model)
        with torch.no_grad():
            posterior = model_list_gp.posterior(train_x_raw)
        output_variables = [
            DistributionVariable(name="output1"),
            DistributionVariable(name="output2"),
        ]
        gp_lume_model = GPModel(
            model=model_list_gp,
            input_variables=input_variables,
            output_variables=output_variables,
        )
        output_dict = gp_lume_model.evaluate(input_dict)
        assert gp_lume_model.get_input_size() == 1
        assert gp_lume_model.get_output_size() == 2
        assert len(output_dict) == 2
        assert torch.allclose(output_dict["output1"].mean, posterior.mean[:, 0])
        assert torch.allclose(output_dict["output1"].variance, posterior.variance[:, 0])
        assert torch.allclose(output_dict["output2"].mean, posterior.mean[:, 1])
        assert torch.allclose(output_dict["output2"].variance, posterior.variance[:, 1])

        # Wrap a multi-output MTGP.
        model_list_gp = ModelListGP(model2)
        with torch.no_grad():
            posterior = model_list_gp.posterior(train_x_raw)
        gp_lume_model = GPModel(
            model=model_list_gp,
            input_variables=input_variables,
            output_variables=output_variables,
        )
        output_dict = gp_lume_model.evaluate(input_dict)
        assert gp_lume_model.get_input_size() == 1
        assert gp_lume_model.get_output_size() == 2
        assert len(output_dict) == 2
        assert torch.allclose(output_dict["output1"].mean, posterior.mean[:, 0])
        assert torch.allclose(output_dict["output1"].variance, posterior.variance[:, 0])
        assert torch.allclose(output_dict["output2"].mean, posterior.mean[:, 1])
        assert torch.allclose(output_dict["output2"].variance, posterior.variance[:, 1])

        # Mix of multi-output and single-output MTGPs.
        model_list_gp = ModelListGP(model, model2, deepcopy(model))
        with torch.no_grad():
            posterior = model_list_gp.posterior(train_x_raw)
        output_variables = [
            DistributionVariable(name="output1"),
            DistributionVariable(name="output2"),
            DistributionVariable(name="output3"),
            DistributionVariable(name="output4"),
        ]
        gp_lume_model = GPModel(
            model=model_list_gp,
            input_variables=input_variables,
            output_variables=output_variables,
        )
        output_dict = gp_lume_model.evaluate(input_dict)
        assert gp_lume_model.get_input_size() == 1
        assert gp_lume_model.get_output_size() == 4
        assert len(output_dict) == 4
        assert torch.allclose(output_dict["output1"].mean, posterior.mean[:, 0])
        assert torch.allclose(output_dict["output1"].variance, posterior.variance[:, 0])
        assert torch.allclose(output_dict["output2"].mean, posterior.mean[:, 1])
        assert torch.allclose(output_dict["output2"].variance, posterior.variance[:, 1])
        assert torch.allclose(output_dict["output3"].mean, posterior.mean[:, 2])
        assert torch.allclose(output_dict["output3"].variance, posterior.variance[:, 2])
        assert torch.allclose(output_dict["output4"].mean, posterior.mean[:, 3])
        assert torch.allclose(output_dict["output4"].variance, posterior.variance[:, 3])

    def test_transformers(
        self,
        create_single_task_gp,
        create_single_task_gp_w_transform,
        single_task_gp_model_kwargs,
    ):
        model, test_x = create_single_task_gp
        model_tf, _, input_transformer, output_transformer = (
            create_single_task_gp_w_transform
        )

        # Test with ModelListGP
        # Test with passing a list of transformers
        model_list_gp = ModelListGP(model, model)
        output_variables = [
            DistributionVariable(name="output1"),
            DistributionVariable(name="output2"),
        ]
        lume_model = GPModel(
            model=model_list_gp,
            input_variables=single_task_gp_model_kwargs["input_variables"],
            output_variables=output_variables,
            input_transformers=[input_transformer],
            output_transformers=[output_transformer],
        )
        # Predict with original model
        test_x_tf = input_transformer.transform(test_x)
        original_pred = model_list_gp.posterior(test_x_tf)
        original_mean = output_transformer.untransform(original_pred.mean)[0]
        original_variance = (
            abs(output_transformer.stdvs.squeeze(0)) ** 2 * original_pred.variance
        )
        # Predict with GPModel
        lume_pred = lume_model.evaluate({"input": test_x})
        assert lume_model.get_input_size() == 1
        assert lume_model.get_output_size() == 2
        assert len(lume_pred) == 2
        assert torch.allclose(original_mean[:, 0], lume_pred["output1"].mean)
        assert torch.allclose(original_variance[:, 0], lume_pred["output1"].variance)
        assert torch.allclose(original_mean[:, 1], lume_pred["output2"].mean)
        assert torch.allclose(original_variance[:, 1], lume_pred["output2"].variance)

        # Test with botorch transformers set as attributes
        # this should ignore any transformers passed to the GPModel
        model_list_gp = ModelListGP(model_tf, model_tf)
        output_variables = [
            DistributionVariable(name="output1"),
            DistributionVariable(name="output2"),
        ]
        lume_model = GPModel(
            model=model_list_gp,
            input_variables=single_task_gp_model_kwargs["input_variables"],
            output_variables=output_variables,
            input_transformers=[input_transformer],
            output_transformers=[output_transformer],
        )

        # Predict with original model
        # We have to transform twice to match the lume-model behavior in this case
        test_x_tf = input_transformer.transform(test_x)
        original_pred = model_list_gp.posterior(test_x_tf)
        original_mean = output_transformer.untransform(original_pred.mean)[0]
        original_variance = (
            abs(output_transformer.stdvs.squeeze(0)) ** 2 * original_pred.variance
        )
        # Predict with GPModel
        lume_pred = lume_model.evaluate({"input": test_x})
        assert lume_model.get_input_size() == 1
        assert lume_model.get_output_size() == 2
        assert len(lume_pred) == 2
        assert torch.allclose(original_mean[:, 0], lume_pred["output1"].mean)
        assert torch.allclose(original_variance[:, 0], lume_pred["output1"].variance)
        assert torch.allclose(original_mean[:, 1], lume_pred["output2"].mean)
        assert torch.allclose(original_variance[:, 1], lume_pred["output2"].variance)

        # Dump and load the model
        lume_model.dump("test.yml")
        loaded_model = GPModel("test.yml")
        lume_pred_loaded = loaded_model.evaluate({"input": test_x})
        assert torch.allclose(
            lume_pred["output1"].mean, lume_pred_loaded["output1"].mean
        )
        assert torch.allclose(
            lume_pred["output1"].variance, lume_pred_loaded["output1"].variance
        )
        assert torch.allclose(
            lume_pred["output2"].mean, lume_pred_loaded["output2"].mean
        )
        assert torch.allclose(
            lume_pred["output2"].variance, lume_pred_loaded["output2"].variance
        )
        os.remove("test.yml")
        os.remove("test_model.pt")
        os.remove("test_input_transformers_0.pt")
        os.remove("test_output_transformers_0.pt")
