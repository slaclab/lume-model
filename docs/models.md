# Models

::: lume_model.base
    options:
        members:
            - LUMEBaseModel
            - process_torch_module
            - model_kwargs_from_dict
            - parse_config
            - json_dumps
            - json_loads
            - recursive_serialize
            - recursive_deserialize


::: lume_model.models.torch_model
    options:
        members:
            - TorchModel
            - InputDictModel

::: lume_model.models.torch_module
    options:
        members:
            - TorchModule
