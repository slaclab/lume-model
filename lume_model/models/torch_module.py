import os
import json
import yaml
import inspect
from typing import Union, Any

import torch

from lume_model.base import parse_config, recursive_serialize
from lume_model.models.torch_model import TorchModel
from lume_model.mlflow_utils import register_model


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
            raise ValueError(
                "Either a YAML string has to be given or model has to be defined."
            )
        super().__init__()
        if len(args) == 1:
            if not all(v is None for v in [model, input_order, output_order]):
                raise ValueError(
                    "Cannot specify YAML string and keyword arguments for TorchModule init."
                )
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
        save_jit: bool = False,
    ) -> str:
        """Serializes the object and returns a YAML formatted string defining the TorchModule instance.

        Args:
            base_key: Base key for serialization.
            file_prefix: Prefix for generated filenames.
            save_models: Determines whether models are saved to file.
            save_jit: Determines whether the structure of the model is saved as TorchScript

        Returns:
            YAML formatted string defining the TorchModule instance.
        """
        d = {}
        for k, v in inspect.signature(TorchModule.__init__).parameters.items():
            if k not in ["self", "args", "model"]:
                d[k] = getattr(self, k)
        output = json.loads(
            json.dumps(
                recursive_serialize(d, base_key, file_prefix, save_models, save_jit)
            )
        )
        model_output = json.loads(
            self._model.to_json(
                base_key=base_key,
                file_prefix=file_prefix,
                save_models=save_models,
                save_jit=save_jit,
            )
        )
        output["model"] = model_output
        # create YAML formatted string
        s = yaml.dump(
            {"model_class": self.__class__.__name__} | output,
            default_flow_style=None,
            sort_keys=False,
        )
        return s

    def dump(
        self,
        file: Union[str, os.PathLike],
        save_models: bool = True,
        base_key: str = "",
        save_jit: bool = False,
    ):
        """Returns and optionally saves YAML formatted string defining the model.

        Args:
            file: File path to which the YAML formatted string and corresponding files are saved.
            base_key: Base key for serialization.
            save_models: Determines whether models are saved to file.
            save_jit : Whether the model is saved using just in time pytorch method
        """
        file_prefix = os.path.splitext(file)[0]
        with open(file, "w") as f:
            f.write(
                self.yaml(
                    save_models=save_models,
                    base_key=base_key,
                    file_prefix=file_prefix,
                    save_jit=save_jit,
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
            [y_model[output_name].unsqueeze(-1) for output_name in self.output_order],
            dim=-1,
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

    def register_to_mlflow(
        self,
        artifact_path: str,
        registered_model_name: str | None = None,
        tags: dict[str, Any] | None = None,
        version_tags: dict[str, Any] | None = None,
        alias: str | None = None,
        run_name: str | None = None,
        log_model_dump: bool = True,
        save_jit: bool = False,
        **kwargs,
    ):
        """
        Registers the model to MLflow if mlflow is installed. Each time this function is called, a new version
        of the model is created. The model is saved to the tracking server or local directory, depending on the
        MLFLOW_TRACKING_URI.

        If no tracking server is set up, data and artifacts are saved directly under your current directory. To set up
        a tracking server, set the environment variable MLFLOW_TRACKING_URI, e.g. a local port/path. See
        https://mlflow.org/docs/latest/getting-started/intro-quickstart/ for more info.

        Args:
            artifact_path: Path to store the model in MLflow.
            registered_model_name: Name of the registered model in MLflow. Optional.
            tags: Tags to add to the MLflow model. Optional.
            version_tags: Tags to add to this MLflow model version. Optional.
            alias: Alias to add to this MLflow model version. Optional.
            run_name: Name of the MLflow run. Optional.
            log_model_dump: Whether to log the model dump files as artifacts. Optional.
            save_jit: Whether to save the model as TorchScript when calling model.dump, if log_model_dump=True. Optional.
            **kwargs: Additional arguments for mlflow.pyfunc.log_model.

        Returns:
            Model info metadata, mlflow.models.model.ModelInfo.
        """
        return register_model(
            self,
            artifact_path,
            registered_model_name,
            tags,
            version_tags,
            alias,
            run_name,
            log_model_dump,
            save_jit,
            **kwargs,
        )


class PriorModel(torch.nn.Module):
    """
    Prior model for Bayesian optimization.
    This module wraps a LUME model and manages the seperation between control variables
    (optimized by Xopt) and fixed variables (measured from the machine). It also maintains
    an efficient buffer of fixed variables that is updated periodically.
    The prior modek is used as a mean function in Gaussian process models to incorporate
    physics knowledge from the LUME surrogate model into the Bayesian optimization process.

    Args:
        model (TorchModule): LUME model that takes all input variables and produces outputs.
            The model's input order is obtained via model.input_variables.
        fixed_variables (dict): Dictionary mapping PV names to their initial measured values
            for all non-control variables. Keys should be PV names (str), values should be
            floats. These represent the initial state of variables not being optimized.

    Attributes:
        model (TorchModule): The LUME surrogate model.
        all_inputs (list): Ordered list of all input variable names from the LUME model.
        control_variables (list): List of control variable names, derived as
            all_inputs - fixed_variables.
        input_buffer (torch.Tensor): 1D tensor storing the current values of all inputs.
            Shape: (n_total_inputs,). This is updated when fixed variables change.
        control_indices (torch.Tensor): 1D tensor of indices for control variables in the
            full input tensor. Shape: (n_control_vars,). Used for fast indexing.
        fixed_indices (list): List of indices for fixed variables in the full input tensor.
    """

    def __init__(self, model: TorchModule, fixed_variables):
        super(PriorModel, self).__init__()
        self.model = model
        self.all_inputs = list(model.input_order)
        self.control_variables = [
            pv for pv in self.all_inputs if pv not in fixed_variables
        ]

        # Create a buffer tensor to store the full input template
        # This is updated ONCE when fixed variables change, not on every forward call
        self.register_buffer("input_buffer", torch.zeros(len(self.all_inputs)))

        # Pre-compute indices for fast lookup (computed once, used many times)
        self.control_indices = torch.tensor(
            [self.all_inputs.index(var) for var in self.control_variables],
            dtype=torch.long,
        )
        self.fixed_indices = [
            self.all_inputs.index(var) for var in fixed_variables.keys()
        ]

        # Initialize buffer with fixed variables
        self.update_fixed_variables(fixed_variables)

        print("PriorModel initialized:")
        print(f"  Total inputs (from model): {len(self.all_inputs)}")
        print(f"  Fixed variables: {len(self.fixed_indices)}")
        print(f"  Control variables (derived): {len(self.control_variables)}")
        print(f"  Control variables: {self.control_variables}")
        print(f"  Control indices: {self.control_indices}")

    def update_fixed_variables(self, fixed_variables):
        """
        Update the buffer with new fixed variable values.

        This method directly updates the input_buffer tensor with new values for
        fixed variables. It should be called when fixed variable measurements change.

        Args:
            fixed_variables (dict): Dictionary mapping PV names to their new measured values.
                Keys should be PV names (str) that exist in self.all_inputs and are NOT
                control variables. Values should be floats.

        Returns:
            None. Updates self.input_buffer in-place.
        """
        for var_name, value in fixed_variables.items():
            idx = self.all_inputs.index(var_name)
            self.input_buffer[idx] = value

    def update_from_data(self, data_row):
        """
        Update fixed variables from a data row (e.g., from X.data).

        Args:
            data_row: pandas Series or dict containing measured values

        Returns:
            dict: Dictionary of {pv_name: value} for the fixed variables that were updated.
        """
        # Extract only the fixed (non-control) variables
        measured_fixed_values = {
            pv: data_row[pv]
            for pv in self.all_inputs
            if pv not in self.control_variables
        }

        # Update the buffer
        self.update_fixed_variables(measured_fixed_values)

        print(f"   Updated buffer with {len(measured_fixed_values)} fixed variables")

        return measured_fixed_values

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass through the LUME model with control and fixed variables

        Args:
            x (torch.Tensor): Tensor containing only control variable values.
                Can have arbitrary batch dimensions.
                The last dimension must match len(self.control_variables).

        Returns:
            torch.Tensor: Output from the LUME model. Shape depends on the model's
                output structure and the input batch dimensions.
        """
        batch_shape = x.shape[
            :-1
        ]  # Get batch shape (everything except the last dimension)

        # Expand buffer to match batch dimensions
        expanded_buffer = self.input_buffer.view(*([1] * len(batch_shape)), -1).expand(
            *batch_shape, -1
        )

        # Clone to make it writable
        full_input = expanded_buffer.clone()

        # Scatter control values into the full input tensor
        # scatter_(dim, index, src)
        # We want to scatter along the last dimension
        indices_expanded = self.control_indices.view(
            *([1] * len(batch_shape)), -1
        ).expand(*batch_shape, -1)

        full_input.scatter_(dim=-1, index=indices_expanded, src=x)

        # Call LUME model
        return self.model(full_input)
