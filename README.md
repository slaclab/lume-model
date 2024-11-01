# LUME-model

LUME-model holds data structures used in the LUME modeling toolset. Variables and models built using LUME-model will be compatible with other tools. LUME-model uses [pydantic](https://pydantic-docs.helpmanual.io/) models to enforce typed attributes upon instantiation.

## Requirements

* Python >= 3.9
* pydantic
* numpy

## Install

LUME-model can be installed with conda using the command:

``` $ conda install lume-model -c conda-forge ```

## Developer

A development environment may be created using the packaged `dev-environment.yml` file.

```
conda env create -f dev-environment.yml
```

## Variables

The LUME-model variables are intended to enforce requirements for input and output variables by variable type. Current variable implementations are scalar (float) or image (numpy array) type.

Example of minimal implementation of scalar input and output variables:

```python
from lume_model.variables import ScalarInputVariable, ScalarOutputVariable

input_variable = ScalarInputVariable(name="test_input", default=0.1, value_range=[1, 2])
output_variable = ScalarOutputVariable(name="test_output")
```

Example of minimal implementation of image input and output variables:

```python
from lume_model.variables import ImageInputVariable, ImageOutputVariable
import numpy as np

input_variable = ImageInputVariable(
    name="test_input",
    default= np.array([[1, 2,], [3, 4]]),
    value_range=[1, 10],
    axis_labels=["count_1", "count_2"],
    x_min=0,
    y_min=0,
    x_max=5,
    y_max=5,
)

output_variable = ImageOutputVariable(
    name="test_output",
    axis_labels=["count_1", "count_2"],
)
```

All input variables may be made into constants by passing the `is_constant=True` keyword argument. Value assingments on these constant variables will raise an error message.

## Models

LUME-model model classes are intended to guide user development while allowing for flexibility and customizability. The base class `lume_model.models.BaseModel` is used to enforce LUME tool compatable classes for the execution of trained models. For this case, model loading and execution should be organized into class methods.

Model Requirements:

* input_variables, output_variables: LUME-model input and output variables are required for use with lume-epics tools. The user can optionally define these as class attributes or design the subclass so that these are passed during initialization . Names of all variables must be unique in order to be served using the EPICS tools. A utility function for saving these variables, which also enforces the uniqueness constraint, is provided (lume_model.utils.save_variables).
* evaluate: The evaluate method is called by the serving model. Subclasses must implement the method, accepting a list of input variables and returning a list of the model's output variables with value attributes updated based on model execution.

Example model implementation:

```python
from lume_model.models import BaseModel

class ExampleModel(BaseModel):
    input_variables = {
        "input1": ScalarInputVariable(name="input1", default=1, range=[0.0, 5.0]),
        "input2": ScalarInputVariable(name="input2", default=2, range=[0.0, 5.0]),
    }

    output_variables = {
        "output1": ScalarOutputVariable(name="output1"),
        "output2": ScalarOutputVariable(name="output2"),
    }

    def evaluate(self, input_variables):

        self.input_variables = {
            variable.name: variable for variable in input_variables
        }

        self.output_variables["output1"].value = (
            self.input_variables["input1"].value * 2
        )
        self.output_variables["output2"].value = (
            self.input_variables["input2"].value * 2
        )

        # return inputs * 2
        return list(self.output_variables.values())
```

## Configuration files

Models and variables may be constructed using a yaml configuration file. The configuration file consists of three sections:

* model (optional, can alternatively pass a custom model class into the `model_from_yaml` method)
* input_variables
* output_variables

The model section is used for the initialization of model classes. The `model_class` entry is used to specify the model class to initialize. The `model_from_yaml` method will attempt to import the specified class. Additional model-specific requirements may be provided. These requirements will be checked before model construction. Model keyword arguments may be passed via the config file or with the function kwarg `model_kwargs`. All models are assumed to accept `input_variables` and `output_variables` as keyword arguments.

The below example outlines the specification for a model compatible with the `lume-model` keras/tensorflow toolkit.

```yaml
model:
    model_class: lume_model.keras.KerasModel
    requirements:
      tensorflow: 2.3.1
    args:
      model_file: examples/files/iris_model.h5
    output_format:
        type: softmax

```

Variables are constructed the minimal data requirements for inputs/outputs.

An example ScalarInputVariable:

```yaml
input_variables:
    SepalLength:
        name: SepalLength
        type: scalar
        default: 4.3
        lower: 4.3
        upper: 7.9

```

For image variables, default values must point to files associated with a default numpy array representation. The file import will be relative to PYTHONPATH.

An example ImageInputVariable:

```yaml
input_variables:
    InputImage:
        name: test
        type: image
        default: examples/files/example_input_image.npy
        range: [0, 100]
        x_min: 0
        x_max: 10
        y_min: 0
        y_max: 10
        axis_labels: ["x", "y"]
        x_min_variable: xmin_pv
        y_min_variable: ymin_pv
        x_max_variable: xmax_pv
        y_max_variable: ymax_pv

```

## PyTorch Toolkit

In the same way as the KerasModel, a PyTorchModel can also be loaded using the `lume_model.utils.model_from_yaml` method, specifying `PyTorchModel` in the `model_class` of the configuration file.

```yaml
model:
  kwargs:
    model_file: /path/to/california_regression.pt
  model_class: lume_model.torch.PyTorchModel
  model_info: path/to/model_info.json
  output_format:
    type: tensor
  requirements:
    torch: 1.12
```

In addition to the model_class, we also specify the path to the pytorch model (saved using `torch.save()`) and additional information about the model through the `model_info.json` file such as the order of the feature names and outputs of the model:

```json
{
    "train_input_mins": [
        0.4999000132083893,
        ...
        -124.3499984741211
    ],
    "train_input_maxs": [
        15.000100135803223,
        ...
        -114.30999755859375
    ],
    "model_in_list": [
        "MedInc",
        ...
        "Longitude"
    ],
    "model_out_list": [
        "MedHouseVal"
    ],
    "loc_in": {
        "MedInc": 0,
        ...
        "Longitude": 7
    },
    "loc_out": {
        "MedHouseVal": 0
    }
}
```

The `output_format` specification indicates which form the outputs of the model's `evaluate()` function should take, which may vary depending on the application. PyTorchModels working with the [LUME-EPICS](https://github.com/slaclab/lume-epics) service will require an `OutputVariable` type, while [Xopt](https://github.com/ChristopherMayes/Xopt) requires either a dictionary of float values or tensors as output.

It is important to note that currently the **transformers are not loaded** into the model when using the `model_from_yaml` method. These need to be created separately and added either:

* to the model's `kwargs` before instantiating

```python
import torch
import json
from lume_model.torch import PyTorchModel

# load the model class and kwargs
with open(f"california_variables.yml","r") as f:
  yaml_model, yaml_kwargs = model_from_yaml(f, load_model=False)

# construct the transformers
with open("normalization.json", "r") as f:
  normalizations = json.load(f)

input_transformer = AffineInputTransform(
    len(normalizations["x_mean"]),
    coefficient=torch.tensor(normalizations["x_scale"]),
    offset=torch.tensor(normalizations["x_mean"]),
)
output_transformer = AffineInputTransform(
    len(normalizations["y_mean"]),
    coefficient=torch.tensor(normalizations["y_scale"]),
    offset=torch.tensor(normalizations["y_mean"]),
)

model_kwargs["input_transformers"] = [input_transformer]
model_kwargs["output_transformers"] = [output_transformer]

model = PyTorchModel(**model_kwargs)
```

* using the setters for the transformer attributes in the model.

```python
# load the model
with open("california_variables.yml", "r") as f:
  model = model_from_yaml(f, load_model=True)

# construct the transformers
with open("normalization.json", "r") as f:
  normalizations = json.load(f)

input_transformer = AffineInputTransform(
    len(normalizations["x_mean"]),
    coefficient=torch.tensor(normalizations["x_scale"]),
    offset=torch.tensor(normalizations["x_mean"]),
)
output_transformer = AffineInputTransform(
    len(normalizations["y_mean"]),
    coefficient=torch.tensor(normalizations["y_scale"]),
    offset=torch.tensor(normalizations["y_mean"]),
)

# use the model's setter to add the transformers. Here we use a tuple
# to tell the setter where in the list the transformer should be inserted.
# In this case because we only have one, we add them at the beginning
# of the lists.
model.input_transformers = (input_transformer, 0)
model.output_transformers = (output_transformer, 0)
```
