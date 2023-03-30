try:
    from .model import PyTorchModel
    from .prior_mean import CustomMean
except ModuleNotFoundError:
    from model import PyTorchModel
    from prior_mean import CustomMean
