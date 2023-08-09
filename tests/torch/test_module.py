import random
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import pytest

try:
    import torch
    from botorch.models import SingleTaskGP

    from lume_model.torch import LUMEModule, PyTorchModel
except ImportError:
    pass


"""
Things to Test
--------------
- [x] a tensor goes in, a tensor comes out
- [x] if the feature order for the MeanModule is different to LumeModel,
    we still get the right output
- [x] output tensor is differentiable
- [x] we are able to override the output of the LUMEModule to predict
    different values to those of the NN
- [x] more than one sample can be evaluated in the LUMEModule (e.g. a
    dataset)
- [x] we can pass in and get an output of >2 dimensions:
    - [x] a dataset of multiple features
    - [x] a dataset of individual features
    - [x] a dataset of GP features [e.g. [50,3,8]]
    - [x] an individual sample of multiple features
    - [x] an individual sample of an individial feature
- [x] we can pass in a tensor of features less than the ones that are
    specified in the model and still get an output
- [x] test that it can be called in a GP loop
- [x] predicting multiple outputs in the LUMEModule returns the expected shape
    (.., n_samples, n_outputs)
- [x] on first creation the Model is frozen but if trainable flag set to True,
    the parameters in the underlying torch module become trainable
- [x] calling parameters in the LUMEModule should return the parameters in the
    underlying torch module in the PyTorchModel
- [x] calling .train() or .eval() on the LUMEModule should also affect the 
    pre-trained model
- [x] if requires grad is set to False, none of the parameters update during
    a training loop (and vice versa)
"""

try:
    y_test = torch.tensor(
        [4.063651, 2.7774928, 2.792812], dtype=torch.double, requires_grad=True
    )

    class MultipleLUMEModule(LUMEModule):
        def __init__(
            self,
            model,
            gp_input_names: List[str] = ...,
            gp_outcome_names: List[str] = ...,
            multiple=2,
        ):
            super().__init__(
                model,
                gp_input_names,
                gp_outcome_names,
            )
            self.multiple = multiple

        def manipulate_outcome(self, y_model: Dict[str, torch.Tensor]):
            y_model[f"{str(self.multiple)}_MedHouseVal"] = (
                y_model["MedHouseVal"] * self.multiple
            )
            return y_model

except NameError:
    # if torch isn't installed then we won't be able to inherit from LUMEModel,
    # doing so will throw an error. In this case we don't create the class and
    # let the try/excepts in the tests allow pytest to skip the tests if torch
    # isn't installed.
    pass


def test_differentiable(california_test_x, cal_model):
    custom_mean = LUMEModule(cal_model, cal_model.features, cal_model.outputs)

    input_tensor = deepcopy(california_test_x)
    input_tensor.requires_grad = True

    result = custom_mean(input_tensor)

    assert result.requires_grad
    for row in result:
        row.backward(retain_graph=True)


def test_model_call_single_feature(california_test_x, cal_model):
    custom_mean = LUMEModule(
        cal_model,
        [cal_model.features[0]],
        cal_model.outputs,
    )

    input_tensor = deepcopy(california_test_x[:, 0].unsqueeze(-1))  # shape (3,1)
    result = custom_mean(input_tensor)
    assert tuple(result.size()) == (3,)


def test_model_call_single_feature_bad_shape(california_test_x, cal_model):
    custom_mean = LUMEModule(
        cal_model,
        [cal_model.features[0]],
        cal_model.outputs,
    )

    input_tensor = deepcopy(california_test_x[:, 0])  # shape (3,)
    with pytest.raises(ValueError) as e:
        custom_mean(input_tensor)
        assert (
            str(e)
            == f"""Expected input dim to be at least 2 ([n_samples, n_features]), received: (3,)"""
        )


def test_model_call_single_datapoint(california_test_x, cal_model):
    custom_mean = LUMEModule(
        cal_model,
        cal_model.features,
        cal_model.outputs,
    )

    input_tensor = deepcopy(california_test_x[0, :]).unsqueeze(0)  # shape (1,8)

    result = custom_mean(input_tensor)

    assert tuple(result.size()) == ()


def test_model_call(california_test_x, cal_model):
    custom_mean = LUMEModule(
        cal_model,
        cal_model.features,
        cal_model.outputs,
    )

    input_tensor = deepcopy(california_test_x[0]).reshape(1, -1)

    result = custom_mean(input_tensor)

    assert tuple(result.size()) == ()
    assert torch.isclose(result, y_test[0])


def test_model_shuffled_inputs(california_test_x, cal_model):
    # shuffle the gaussian process features and the values associated with them
    shuffled_x = list(zip(cal_model.features, california_test_x[0]))
    random.shuffle(shuffled_x)
    gp_features = [shuffled_x[i][0] for i in range(len(shuffled_x))]
    gp_inputs = torch.tensor(
        [shuffled_x[i][1] for i in range(len(shuffled_x))]
    ).reshape(1, -1)
    gp_outputs = deepcopy(cal_model.outputs)

    custom_mean = LUMEModule(cal_model, gp_features, gp_outputs)

    result = custom_mean(gp_inputs)

    assert tuple(result.size()) == ()
    # despite the shuffled inputs, the output values should be the same
    assert torch.isclose(result, y_test[0])


def test_model_call_modified_output(california_test_x, cal_model):
    multiple = 2
    output_order = deepcopy(cal_model.outputs)
    output_order.append(f"{str(multiple)}_MedHouseVal")

    custom_mean = MultipleLUMEModule(
        cal_model, cal_model.features, output_order, multiple=multiple
    )

    input_tensor = deepcopy(california_test_x[0]).reshape(1, -1)

    result = custom_mean(input_tensor)

    assert tuple(result.size()) == (2,)
    assert torch.isclose(result[0], y_test[0])
    assert torch.isclose(result[1], multiple * y_test[0])


def test_model_call_multiple_outputs(california_test_x, cal_model):
    multiple = 2
    output_order = deepcopy(cal_model.outputs)
    output_order.append(f"{str(multiple)}_MedHouseVal")

    custom_mean = MultipleLUMEModule(
        cal_model, cal_model.features, output_order, multiple=multiple
    )

    input_tensor = deepcopy(california_test_x)

    result = custom_mean(input_tensor)

    assert tuple(result.size()) == (3, 2)
    assert all(
        torch.isclose(
            result[:, 0],
            y_test,
        )
    )
    assert all(torch.isclose(result[:, 1], multiple * y_test))


def test_model_call_dataset(california_test_x, cal_model):
    custom_mean = LUMEModule(
        cal_model,
        cal_model.features,
        cal_model.outputs,
    )

    result = custom_mean(california_test_x)
    assert tuple(result.size()) == (3,)
    assert all(torch.isclose(result, y_test))


def test_california_housing_model_multi_dim_tensor(
    california_test_x,
    cal_model,
):
    # when using the model within a custom mean, we might get some data that
    # comes through in the dictionary as shape [b, n, m] where b is the batch
    # number, n is the number of data points and m is the number of features,
    # the model should be able to cope with these as well
    batch_size = 5
    test_data = california_test_x.unsqueeze(0).repeat((batch_size, 1, 1))

    custom_mean = LUMEModule(
        cal_model,
        cal_model.features,
        cal_model.outputs,
    )
    result = custom_mean(test_data)

    assert tuple(result.shape) == (batch_size, 3)
    assert all(torch.isclose(result[0, :], y_test))


def test_california_housing_model_multi_dim_tensor_multi_output(
    california_test_x,
    cal_model,
):
    # when using the model within a custom mean, we might get some data that
    # comes through in the dictionary as shape [b, n, m] where b is the batch
    # number, n is the number of data points and m is the number of features,
    # the model should be able to cope with these as well
    batch_size = 5
    multiple = 2
    test_data = california_test_x.unsqueeze(0).repeat((batch_size, 1, 1))

    output_order = deepcopy(cal_model.outputs)
    output_order.append(f"{str(multiple)}_MedHouseVal")

    custom_mean = MultipleLUMEModule(
        cal_model, cal_model.features, output_order, multiple=multiple
    )

    result = custom_mean(test_data)

    assert tuple(result.shape) == (batch_size, 3, 2)

    assert all(torch.isclose(result[0, :, 0], y_test))
    assert all(torch.isclose(result[0, :, 1], multiple * y_test))


def test_gp_loop(california_test_x, cal_model):
    custom_mean = LUMEModule(
        cal_model,
        cal_model.features,
        cal_model.outputs,
    )
    train_x = california_test_x.double()
    train_y = custom_mean(train_x).unsqueeze(-1)
    gp = SingleTaskGP(
        train_x,  # pretend we have three data points
        train_y,
        mean_module=custom_mean,
    )
    post = gp.posterior(train_x)
    mean = post.mean.squeeze()
    assert tuple(mean.shape) == (3,)
    # before training the GP should just predict the custom mean value
    # for the posterior
    assert all(torch.isclose(mean, y_test))


def test_trainable(cal_model):
    custom_mean = LUMEModule(
        cal_model,
        cal_model.features,
        cal_model.outputs,
    )
    # by default, the model is set to be not trainable
    for param in custom_mean.parameters():
        assert param.requires_grad is False
    for param in custom_mean._model.model.parameters():
        assert param.requires_grad is False

    # if we enforce trainability in the module this should filter
    # down to the pytorch model within the LUMEModule
    custom_mean.requires_grad_(True)
    for param in custom_mean.parameters():
        assert param.requires_grad is True
    for param in custom_mean._model.model.parameters():
        assert param.requires_grad is True


def test_parameters(cal_model):
    custom_mean = LUMEModule(
        cal_model,
        cal_model.features,
        cal_model.outputs,
    )
    assert len(list(custom_mean.parameters())) == len(
        list(custom_mean._model.model.parameters())
    )
    params = [
        a == b
        for a, b in zip(custom_mean.parameters(), custom_mean._model.model.parameters())
    ]
    for param in params:
        assert torch.all(param)


def test_train_eval(cal_model):
    custom_mean = LUMEModule(
        cal_model,
        cal_model.features,
        cal_model.outputs,
    )

    # check that if we set the LUMEModule as eval() or training(), the underlying
    # model changes as well
    assert custom_mean.training is False
    assert custom_mean._model.model.training is False
    custom_mean.train()
    assert custom_mean.training is True
    assert custom_mean._model.model.training is True
    custom_mean.eval()
    assert custom_mean.training is False
    assert custom_mean._model.model.training is False


def test_non_trainable_loop(california_test_x, cal_model):
    custom_mean = LUMEModule(
        cal_model,
        cal_model.features,
        cal_model.outputs,
    )
    custom_mean.requires_grad_(False)

    original_model = deepcopy(str(custom_mean._model.model.state_dict()))
    base_model_dict = {
        k.replace("base_model.", ""): v for k, v in custom_mean.state_dict().items() if
        not k.startswith(("input_transformer", "output_transformer"))
    }
    base_model = deepcopy(str(OrderedDict(base_model_dict)))

    optimizer = torch.optim.Adam(custom_mean._model.model.parameters())
    loss_fn = torch.nn.MSELoss()
    for _ in range(2):
        custom_mean.train()
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = custom_mean(california_test_x)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, y_test)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
    post_training_original_model = deepcopy(str(custom_mean._model.model.state_dict()))
    post_training_base_model_dict = {
        k.replace("base_model.", ""): v for k, v in custom_mean.state_dict().items() if
        not k.startswith(("input_transformer", "output_transformer"))
    }
    post_training_base_model = deepcopy(str(OrderedDict(post_training_base_model_dict)))

    assert (
        post_training_original_model
        == post_training_base_model
        == original_model
        == base_model
    )


def test_trainable_loop(california_test_x, cal_model):
    custom_mean = LUMEModule(
        cal_model,
        cal_model.features,
        cal_model.outputs,
    )
    custom_mean.requires_grad_(True)

    original_model = deepcopy(str(custom_mean._model.model.state_dict()))
    base_model_dict = {
        k.replace("base_model.", ""): v for k, v in custom_mean.state_dict().items() if
        not k.startswith(("input_transformer", "output_transformer"))
    }
    base_model = deepcopy(str(OrderedDict(base_model_dict)))

    optimizer = torch.optim.Adam(custom_mean._model.model.parameters())
    loss_fn = torch.nn.MSELoss()
    for _ in range(2):
        custom_mean.train()
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = custom_mean(california_test_x)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, y_test)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
    post_training_original_model = deepcopy(str(custom_mean._model.model.state_dict()))
    post_training_base_model_dict = {
        k.replace("base_model.", ""): v for k, v in custom_mean.state_dict().items() if
        not k.startswith(("input_transformer", "output_transformer"))
    }
    post_training_base_model = deepcopy(str(OrderedDict(post_training_base_model_dict)))

    assert post_training_original_model == post_training_base_model
    assert original_model == base_model
    assert post_training_original_model != original_model
    assert post_training_base_model != base_model
