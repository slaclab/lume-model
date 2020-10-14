# Model development with the Keras/tensorflow toolkit

At present, only the tensorflow backend is supported for this toolkit.

The `KerasModel` packaged in the toolkit will be compatible with models saved using the `keras.save_model()` method.

## Development requirements:
- The model must be trained using the custom scaling layers provided in `lume_model.keras.layers` OR using preprocessing layers packaged with Keras OR the custom layers must be defined during build and made accessible during loading by the user. Custom layers are not supported out-of-the box by this toolkit.
- The keras model must use named input layers such that the model will accept a dictionary input OR the `KerasModel` must be subclassed and the `format_input` and `format_output` member functions must be overwritten with proper formatting of model input from a dictionary mapping input variable names to values and proper output parsing into a dictionary, respectively.

## Configuration file
The KerasModel can be instantiated using the utility function `lume_model.utils.model_from_yaml` method.

KerasModel can be specified in the `model_class` of the model configuration.
```yaml
model:
    model_class: lume_model.keras.KerasModel
```

Custom parsing will require a custom model class.
