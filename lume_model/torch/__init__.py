try:
    from .model import PyTorchModel
    from .module import LUMEModule
except ModuleNotFoundError:
    from model import PyTorchModel
    from lume_model.torch.module import LUMEModule
