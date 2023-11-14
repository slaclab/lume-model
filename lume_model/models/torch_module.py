import os
import json
import yaml
import inspect
from typing import Union

import torch

from lume_model.base import parse_config, recursive_serialize
from lume_model.models.torch_model import TorchModel


class TorchModule(torch.nn.Module):
    """Wrapper to allow a LUME TorchModel to be used like a torch.nn.Module.

    As the base model within the TorchModel is assumed to be fixed during instantiation,
    so is the TorchModule.
    """
    def __init__(
        self,
        *args,
        model: TorchModel = None,
        input_order: list[str] = None,
        output_order: list[str] = None,
    ):
        """Initializes TorchModule.

        Args:
            *args: Accepts a single argument which is the model configuration as dictionary, YAML or JSON
              formatted string or file path.

        Keyword Args:
            model: The TorchModel instance to wrap around. If config is None, this has to be defined.
            input_order: Input names in the order they are passed to the model. If None, the input order of the
              TorchModel is used.
            output_order: Output names in the order they are returned by the model. If None, the output order of
              the TorchModel is used.
        """
        if all(arg is None for arg in [*args, model]):
            raise ValueError("Either a YAML string has to be given or model has to be defined.")
        super().__init__()
        if len(args) == 1:
            if not all(v is None for v in [model, input_order, output_order]):
                raise ValueError("Cannot specify YAML string and keyword arguments for TorchModule init.")
            model_fields = {f"model.{k}": v for k, v in TorchModel.model_fields.items()}
            kwargs = parse_config(args[0], model_fields)
            kwargs["model"] = TorchModel(kwargs["model"])
            self.__init__(**kwargs)
        elif len(args) > 1:
            raise ValueError(
                "Arguments to TorchModule must be either a single YAML string or keyword arguments."
            )
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
            base_key: str = "",
            file_prefix: str = "",
            save_models: bool = False,
    ) -> str:
        """Serializes the object and returns a YAML formatted string defining the TorchModule instance.

        Args:
            base_key: Base key for serialization.
            file_prefix: Prefix for generated filenames.
            save_models: Determines whether models are saved to file.

        Returns:
            YAML formatted string defining the TorchModule instance.
        """
        d = {}
        for k, v in inspect.signature(TorchModule.__init__).parameters.items():
            if k not in ["self", "args", "model"]:
                d[k] = getattr(self, k)
        output = json.loads(
            json.dumps(recursive_serialize(d, base_key, file_prefix, save_models))
        )
        model_output = json.loads(
            self._model.to_json(
                base_key=base_key,
                file_prefix=file_prefix,
                save_models=save_models,
            )
        )
        output["model"] = model_output
        # create YAML formatted string
        s = yaml.dump({"model_class": self.__class__.__name__} | output,
                      default_flow_style=None, sort_keys=False)
        return s

    def dump(
            self,
            file: Union[str, os.PathLike],
            save_models: bool = True,
            base_key: str = "",
    ):
        """Returns and optionally saves YAML formatted string defining the model.

        Args:
            file: File path to which the YAML formatted string and corresponding files are saved.
            base_key: Base key for serialization.
            save_models: Determines whether models are saved to file.
        """
        file_prefix = os.path.splitext(file)[0]
        with open(file, "w") as f:
            f.write(
                self.yaml(
                    save_models=save_models,
                    base_key=base_key,
                    file_prefix=file_prefix,
                )
            )

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
