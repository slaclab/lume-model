import os
import json
import yaml
import inspect
from typing import Union

import torch

from lume_model.base import json_dumps, parse_config
from lume_model.torch.model import TorchModel


class TorchModule(torch.nn.Module):
    """Wrapper to allow a LUME TorchModel to be used like a torch.nn.Module.

    As the base model within the TorchModel is assumed to be fixed during instantiation,
    so is the TorchModule.
    """
    def __init__(
        self,
        config: Union[dict, str] = None,
        *,
        model: TorchModel = None,
        input_order: list[str] = None,
        output_order: list[str] = None,
    ):
        """Initializes TorchModule.

        Args:
            config: Model configuration as dictionary, YAML or JSON formatted string or file path. This overrides
              all other arguments.

        Keyword Args:
            model: The TorchModel instance to wrap around. If config is None, this has to be defined.
            input_order: Input names in the order they are passed to the model. If None, the input order of the
              TorchModel is used.
            output_order: Output names in the order they are returned by the model. If None, the output order of
              the TorchModel is used.
        """
        if all(arg is None for arg in [config, model]):
            raise ValueError("Either config or model has to be defined.")
        super().__init__()
        if config is not None:
            kwargs = parse_config(config)
            kwargs["model"] = TorchModel(kwargs["model"])
            self.__init__(**kwargs)
        else:
            self._model = model
            self._input_order = input_order
            self._output_order = output_order
            self.register_module("base_model", self._model.model)
            for i, input_transformer in enumerate(self._model.input_transformers):
                self.register_module(f"input_transformers_{i}", input_transformer)
            for i, output_transformer in enumerate(self._model.output_transformers):
                self.register_module(f"output_transformers_{i}", output_transformer)
            if not model.model.training:  # TorchModel defines train/eval mode
                self.eval()

    @property
    def model(self):
        return self._model

    @property
    def input_order(self):
        if self._input_order is None:
            return self._model.input_names
        else:
            return self._input_order

    @property
    def output_order(self):
        if self._output_order is None:
            return self._model.output_names
        else:
            return self._output_order

    def forward(self, x: torch.Tensor):
        # input shape: [n_batch, n_samples, n_dim]
        x = self._validate_input(x)
        model_input = self._tensor_to_dictionary(x)
        y_model = self.evaluate_model(model_input)
        y_model = self.manipulate_output(y_model)
        # squeeze for use as prior mean in botorch GPs
        y = self._dictionary_to_tensor(y_model).squeeze()
        return y

    def yaml(
            self,
            file: Union[str, os.PathLike] = None,
            save_models: bool = True,
            base_key: str = "",
    ) -> str:
        """Returns and optionally saves YAML formatted string defining the TorchModule instance.

        Args:
            file: If not None, YAML formatted string is saved to given file path.
            save_models: Determines whether models are saved to file.
            base_key: Base key for serialization.

        Returns:
            YAML formatted string defining the TorchModule instance.
        """
        file_prefix = ""
        if file is not None:
            file_prefix = os.path.splitext(file)[0]
        # get TorchModel config
        d = {}
        for k, v in inspect.signature(TorchModule.__init__).parameters.items():
            if k not in ["self", "config", "model"]:
                d[k] = getattr(self, k)
        config = json.loads(
            json_dumps(d, default=None, base_key=base_key, file_prefix=file_prefix, save_models=save_models)
        )
        model_config = json.loads(
            self._model.json(base_key=base_key, file_prefix=file_prefix, save_models=save_models)
        )
        config["model"] = model_config
        # create YAML formatted string
        s = yaml.dump({"model_class": self.__class__.__name__} | config,
                      default_flow_style=None, sort_keys=False)
        if file is not None:
            with open(file, "w") as f:
                f.write(s)
        return s

    def evaluate_model(self, x: dict[str, torch.Tensor]):
        """Placeholder method to modify model calls."""
        return self._model.evaluate(x)

    def manipulate_output(self, y_model: dict[str, torch.Tensor]):
        """Placeholder method to modify the model output."""
        return y_model

    def _tensor_to_dictionary(self, x: torch.Tensor):
        input_dict = {}
        for idx, input_name in enumerate(self.input_order):
            input_dict[input_name] = x[..., idx].unsqueeze(-1)
        return input_dict

    def _dictionary_to_tensor(self, y_model: dict[str, torch.Tensor]):
        output_tensor = torch.stack(
            [y_model[output_name].unsqueeze(-1) for output_name in self.output_order], dim=-1
        )
        return output_tensor

    @staticmethod
    def _validate_input(x: torch.Tensor) -> torch.Tensor:
        if x.dim() <= 1:
            raise ValueError(
                f"Expected input dim to be at least 2 ([n_samples, n_features]), received: {tuple(x.shape)}"
            )
        else:
            return x
