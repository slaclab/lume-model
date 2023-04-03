try:
    from .model import PyTorchModel
    from .prior_mean import LUMEModule
except ModuleNotFoundError:
    from model import PyTorchModel
    from prior_mean import LUMEModule
