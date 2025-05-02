import os
import warnings
from typing import Dict, Any, Union

from torch import Tensor
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    pass


def register_model(
            lume_model,
            input_dict: dict[str, Union[float, Tensor]],
            artifact_path: str,
            registered_model_name: str | None = None,
            tags: dict[str, Any] | None = None,
            version_tags: dict[str, Any] | None = None,
            alias: str | None = None,
            run_name: str | None = None,
            log_model_dump: bool = True,
            **kwargs
    ) -> mlflow.models.model.ModelInfo:
    """
    Registers the model to MLflow if mlflow is installed. Each time this function is called, a new version
    of the model is created. The model is saved to the tracking server or local directory, depending on the
    MLFLOW_TRACKING_URI.

    If no tracking server is set up, data and artifacts are saved directly under your current directory. To set up
    a tracking server, set the environment variable MLFLOW_TRACKING_URI, e.g. a local port/path. See
    https://mlflow.org/docs/latest/getting-started/intro-quickstart/ for more info.

    Args:
        lume_model: LumeModel to register.
        input_dict: Input dictionary to infer the model signature.
        artifact_path: Path to store the model in MLflow.
        registered_model_name: Name of the registered model in MLflow.
        tags: Tags to add to the MLflow model.
        version_tags: Tags to add to this MLflow model version.
        alias: Alias to add to this MLflow model version.
        run_name: Name of the MLflow run.
        log_model_dump: If True, the model dump is logged to MLflow.
        **kwargs: Additional arguments for mlflow.pytorch.log_model or torch.save.

    Returns:
        Model info metadata, mlflow.models.model.ModelInfo.
    """
    if not HAS_MLFLOW:
        raise ImportError("MLflow is not installed. Cannot register model.")
    if "MLFLOW_TRACKING_URI" not in os.environ:
        warnings.warn(
            "MLFLOW_TRACKING_URI is not set. Data and artifacts will be saved directly under your current directory."
        )

    # Adjust the input_dict to match the expected input format
    # Input must be one of `numpy.ndarray`, `List[numpy.ndarray]`, `Dict[str, numpy.ndarray]` or `pandas.DataFrame`
    input_dict = {
        key: value.numpy()
        for key, value in input_dict.items()
    }

    # Create pyfunc model for MLflow to be able to log/load the model
    pf_model = create_mlflow_model(lume_model)

    # Define the signature of the model
    signature = mlflow.models.infer_signature(input_dict, pf_model.predict([input_dict]))

    # Log the model to MLflow
    with mlflow.start_run(run_name=run_name):
        model_info = mlflow.pyfunc.log_model(
            python_model=pf_model,
            artifact_path=artifact_path,
            signature=signature,
            registered_model_name=registered_model_name,
            **kwargs
        )
        if log_model_dump:
            # Log the model dump to MLflow
            #model_dump_path = os.path.join("model_dump.yml")
            # TODO: pass directory where user wants local dump to, default to working directory
            name = "model_dump"
            lume_model.dump(f"{name}.yml")
            mlflow.log_artifact(f"{name}_model.pt", artifact_path)
            # TODO: arg to add jit saving?
            try:
                mlflow.log_artifact(f"{name}.jit", artifact_path)
            except FileNotFoundError:
                pass
            # TODO: get names of in/out transform files and save them


    if (tags or alias or version_tags) and registered_model_name:
        from mlflow import MlflowClient

        client = MlflowClient()
        # Get the latest version of the registered model that we just registered
        latest_version = client.get_latest_versions(registered_model_name)[0].version

        if tags:
            for key, value in tags.items():
                client.set_registered_model_tag(registered_model_name, key, value)
        if version_tags:
            for key, value in version_tags.items():
                client.set_model_version_tag(
                    registered_model_name,
                    latest_version,
                    key,
                    value
                )
        if alias:
            client.set_registered_model_alias(registered_model_name, alias, latest_version)

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
    def __init__(self, model):
        self.model = model

    def predict(self, model_input: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the model with the given input."""
        # Convert input to the format expected by the model
        # TODO: this isn't very general but type validation in torch modules requires this. May need to adjust.
        model_input = {
            key: Tensor(value)
            for key, value in model_input[0].items()
        }
        return self.model.evaluate(model_input)

    def save_model(self):
        raise NotImplementedError("Save model not implemented")

    def load_model(self):
        raise NotImplementedError("Load model not implemented")

    def get_lume_model(self):
        return self.model