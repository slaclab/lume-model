"""Module for training or retraining a TorchModule."""

from typing import Tuple
import torch
import matplotlib.pyplot as plt


# TODO: add type validation
def train(
    model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    params: list[torch.nn.parameter.Parameter] | None = None,
    lr: float = 1e-3,
    reg: float = 1e-3,
    epochs: int = int(1e3),
    optimizer: str = "adam",
    criterion: str = "mae",
    ckwargs: dict | None = None,
    plot: bool = True,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, list[float]]:
    """Train a PyTorch model on a given dataset.
    Args:
        model (torch.nn.Module): The PyTorch model to train.
        trainloader (torch.utils.data.Dataloader): The iterable batched dataset to train on.
        params (list[torch.nn.parameter.Parameter], optional): List of parameters to optimize.
            If None, all model parameters are used. Defaults to None.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        reg (float, optional): Regularization term for the optimizer. Defaults to 1ec-3.
        epochs (int, optional): Number of epochs to train for. Defaults to 1000.
        optimizer (str, optional): Optimizer to use. Options are "adam" or "sgd". Defaults to "adam".
        criterion (str, optional): Loss function to use. Options are "mse" or "mae". Defaults to "mae".
        ckwargs (dict, optional): Additional keyword arguments for the loss function. Defaults to None.
        plot (bool, optional): Whether to plot the training loss. Defaults to True.
        verbose (bool, optional): Whether to print training progress. Defaults to True.
    Returns:
        tuple: A tuple containing the trained model and a list of training losses.
    Raises:
        ValueError: If an unsupported optimizer or loss function is specified.
    """

    if params is None:
        params = []
        try:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params.append(getattr(model, name))
        except AttributeError:
            # If model does not have named_parameters, use all parameters
            params = list(model.parameters())

    if optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=reg)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=reg)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    if criterion == "mse":
        criterion = torch.nn.MSELoss(**ckwargs)
    elif criterion == "mae":
        criterion = torch.nn.L1Loss(**ckwargs)
    else:
        raise ValueError(f"Unsupported loss function: {criterion}")

    model.train()
    losses = []
    epochs = int(epochs)
    for epoch in range(epochs):
        for i, batch_data in enumerate(trainloader, 0):
            inputs, targets = batch_data
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            if i == 0:
                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(trainloader)}, Loss: {loss.item()}"
                    )
                losses.append(loss.item())

    if plot:
        _ = plot_training_loss(losses, criterion=criterion)

    return model, losses


def create_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> torch.utils.data.Dataset:
    """Create a PyTorch dataset from input and target tensors.

    Args:
        x (torch.Tensor): Input tensor of shape (n_samples, n_features).
        y (torch.Tensor): Target tensor of shape (n_samples,).
        batch_size (int, optional): Batch size for the DataLoader. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory. Defaults to True.

    Returns:
        torch.utils.data.Dataset: A PyTorch dataset containing the input and target tensors.
    """
    dataset = Dataset(x, y)
    batch_size = batch_size or y.shape[0]
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return trainloader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        (
            self.x,
            self.y,
        ) = x, y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        return x, y


def plot_training_loss(
    losses,
    criterion: str | torch.nn.Module = "mae",
    figsize: Tuple[int, int] = (6, 4),
) -> plt.Axes:
    """Plot the training loss over epochs."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(losses, "C0")
    ax.set_xlabel("Epoch")
    criterion = (
        criterion.__class__.__name__
        if isinstance(criterion, torch.nn.Module)
        else criterion.upper()
    )
    ax.set_ylabel(criterion)
    ax.grid(color="gray", linestyle="dashed")
    ax.set_yscale("log")
    fig.tight_layout()
    plt.show()
    return ax
