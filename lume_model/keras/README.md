# Model development with the Keras/tensorflow toolkit

At present, only the tensorflow backend is supported for this toolkit. For use, the metapackage for `lume-model` must be installed. This can be accessed via conda using:

``` $ conda install lume-model-keras -c jrgarrahan```

The `BaseModel` packaged in the toolkit will be compatible with models saved using the `keras.save_model()` method.

Development requirements:
- The model must be trained using the custom scaling layers provided in `lume_model.keras.layers` OR the custom layers must be defined during build and made accessible during loading by the user. No custom layers will be supported out-of-the box by this toolkit.
- The `BaseModel` must be subclassed and the `format_input` and `format_output` class methods must be implemented.
