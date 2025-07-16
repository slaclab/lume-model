import os
import warnings
from typing import Any
from contextlib import nullcontext

from torch import nn

try:
    import mlflow

    HAS_MLFLOW = True
    import logging

    logger = logging.getLogger("mlflow")
    # Set log level to error until annoying signature warnings are fixed
    logger.setLevel(logging.ERROR)
except ImportError:
    pass


def register_model(
    lume_model,
    artifact_path: str,
    registered_model_name: str | None = None,
    tags: dict[str, Any] | None = None,
    version_tags: dict[str, Any] | None = None,
    alias: str | None = None,
    run_name: str | None = None,
    log_model_dump: bool = True,
    save_jit: bool = False,
    **kwargs,
) -> mlflow.models.model.ModelInfo:
    """
    Registers the model to MLflow if mlflow is installed. Each time this function is called, a new version
    of the model is created. The model is saved to the tracking server or local directory, depending on the
    MLFLOW_TRACKING_URI.

    If no tracking server is set up, data and artifacts are saved directly under your current directory. To set up
    a tracking server, set the environment variable MLFLOW_TRACKING_URI, e.g. a local port/path. See
    https://mlflow.org/docs/latest/getting-started/intro-quickstart/ for more info.

    Note that at the moment, this does not log artifacts for custom models other than the YAML dump file.

    Args:
        lume_model: LumeModel to register.
        artifact_path: Path to store the model in MLflow.
        registered_model_name: Name of the registered model in MLflow.
        tags: Tags to add to the MLflow model.
        version_tags: Tags to add to this MLflow model version.
        alias: Alias to add to this MLflow model version.
        run_name: Name of the MLflow run.
        log_model_dump: If True, the model dump is logged to MLflow.
        save_jit: If True, the model is saved as a JIT model when calling model.dump, if log_model_dump=True.
        **kwargs: Additional arguments for mlflow.pyfunc.log_model.

    Returns:
        Model info metadata, mlflow.models.model.ModelInfo.
    """
    if not HAS_MLFLOW:
        raise ImportError("MLflow is not installed. Cannot register model.")
    if "MLFLOW_TRACKING_URI" not in os.environ:
        warnings.warn(
            "MLFLOW_TRACKING_URI is not set. Data and artifacts will be saved directly under your current directory."
        )

    # Log the model to MLflow
    ctx = (
        mlflow.start_run(run_name=run_name)
        if mlflow.active_run() is None
        else nullcontext()
    )
    with ctx:
        if isinstance(lume_model, nn.Module):
            model_info = mlflow.pytorch.log_model(
                pytorch_model=lume_model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        else:
            # Create pyfunc model for MLflow to be able to log/load the model
            pf_model = create_mlflow_model(lume_model)
            model_info = mlflow.pyfunc.log_model(
                python_model=pf_model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                **kwargs,
            )

        if log_model_dump:
            # Log the model dump files to MLflow
            # TODO: pass directory where user wants local dump to, default to working directory
            run_name = mlflow.active_run().info.run_name
            name = registered_model_name or f"{run_name}"

            lume_model.dump(f"{name}.yml", save_jit=save_jit)
            mlflow.log_artifact(f"{name}.yml", artifact_path)
            os.remove(f"{name}.yml")

            from lume_model.models import registered_models

            if type(lume_model) in registered_models:
                # all registered models are torch models at the moment
                # may change in the future
                mlflow.log_artifact(f"{name}_model.pt", artifact_path)
                os.remove(f"{name}_model.pt")
                if save_jit:
                    mlflow.log_artifact(f"{name}_model.jit", artifact_path)
                    os.remove(f"{name}_model.jit")

                # Get and log the input and output transformers
                lume_model = (
                    lume_model._model
                    if isinstance(lume_model, nn.Module)
                    else lume_model
                )
                for i in range(len(lume_model.input_transformers)):
                    mlflow.log_artifact(
                        f"{name}_input_transformers_{i}.pt", artifact_path
                    )
                    os.remove(f"{name}_input_transformers_{i}.pt")
                for i in range(len(lume_model.output_transformers)):
                    mlflow.log_artifact(
                        f"{name}_output_transformers_{i}.pt", artifact_path
                    )
                    os.remove(f"{name}_output_transformers_{i}.pt")

    if (tags or alias or version_tags) and registered_model_name:
        from mlflow import MlflowClient

        client = MlflowClient()
        # Get the latest version of the registered model that we just registered
        latest_version = model_info.registered_model_version

        if tags:
            for key, value in tags.items():
                client.set_registered_model_tag(registered_model_name, key, value)
        if version_tags:
            for key, value in version_tags.items():
                client.set_model_version_tag(
                    registered_model_name, latest_version, key, value
                )
        if alias:
            client.set_registered_model_alias(
                registered_model_name, alias, latest_version
            )

    elif (tags or alias or version_tags) and not registered_model_name:
        warnings.warn(
            "No registered model name provided. Tags and aliases will not be set."
        )

    return model_info


def create_mlflow_model(model) -> mlflow.pyfunc.PythonModel:
    """Creates an MLflow model from the given model."""
    return PyFuncModel(model=model)


class PyFuncModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow model class for LumeModel.
    Uses Pyfunc to define a model that can be saved and loaded with MLflow.

    Must implement the `predict` method.
    """

    # Disable type hint validation for the predict method to avoid annoying warnings
    # since we have type validation in the lume-model itself.
    _skip_type_hint_validation = True

    def __init__(self, model):
        self.model = model

    def predict(self, model_input):
        """Evaluate the model with the given input."""
        return self.model.evaluate(model_input)

    def save_model(self):
        raise NotImplementedError("Save model not implemented")

    def load_model(self):
        raise NotImplementedError("Load model not implemented")

    def get_lume_model(self):
        return self.model
