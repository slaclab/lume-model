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

The lume-model variables are intended to enforce requirements for input and output variables by variable type. For now, only scalar variables (floats) are supported.

Minimal example of scalar input and output variables:

```python
from lume_model.variables import ScalarInputVariable, ScalarOutputVariable

input_variable = ScalarInputVariable(
    name="example_input",
    default=0.1,
    value_range=[0.0, 1.0],
)
output_variable = ScalarOutputVariable(name="example_output")
```

All input variables may be made into constants by passing the `is_constant=True` keyword argument. Value assingments on these constant variables will raise an error message.

## Models

The lume-model base class `lume_model.base.LUMEBaseModel` is intended to guide user development while allowing for flexibility and customizability. It is used to enforce LUME tool compatible classes for the execution of trained models.

Requirements for model classes:

* input_variables: A list defining the input variables for the model. Variable names must be unique. Required for use with lume-epics tools.
* output_variables: A list defining the output variables for the model. Variable names must be unique. Required for use with lume-epics tools.
* evaluate: The evaluate method is called by the serving model. Subclasses must implement this method, accepting and returning a dictionary.

Example model implementation and instantiation:

```python
from lume_model.base import LUMEBaseModel
from lume_model.variables import ScalarInputVariable, ScalarOutputVariable


class ExampleModel(LUMEBaseModel):
    def evaluate(self, input_dict):
        output_dict = {
            "output1": input_dict[self.input_variables[0].name] ** 2,
            "output2": input_dict[self.input_variables[1].name] ** 2,
        }
        return output_dict


input_variables = [
    ScalarInputVariable(name="input1", default=0.1, value_range=[0.0, 1.0]),
    ScalarInputVariable(name="input2", default=0.2, value_range=[0.0, 1.0]),
]
output_variables = [
    ScalarOutputVariable(name="output1"),
    ScalarOutputVariable(name="output2"),
]

m = ExampleModel(input_variables=input_variables, output_variables=output_variables)
```

## Configuration files

Models and variables may be constructed using a YAML configuration file. The configuration file consists of three sections:

* model (optional, can alternatively pass a custom model class into the `model_from_yaml` method)
* input_variables
* output_variables

The model section is used for the initialization of model classes. The `model_class` entry is used to specify the model class to initialize. The `model_from_yaml` method will attempt to import the specified class. Additional model-specific requirements may be provided. These requirements will be checked before model construction. Model keyword arguments may be passed via the config file or with the function kwarg `model_kwargs`. All models are assumed to accept `input_variables` and `output_variables` as keyword arguments.

For example, `m.dump("example_model.yml")` writes the following to file

```yaml
model_class: ExampleModel
input_variables:
  input1:
    variable_type: scalar
    default: 0.1
    is_constant: false
    value_range: [0.0, 1.0]
  input2:
    variable_type: scalar
    default: 0.2
    is_constant: false
    value_range: [0.0, 1.0]
output_variables:
  output1: {variable_type: scalar}
  output2: {variable_type: scalar}
```

and can be loaded by simply passing the file to the model constructor:

```python
from lume_model.base import LUMEBaseModel


class ExampleModel(LUMEBaseModel):
    def evaluate(self, input_dict):
        output_dict = {
            "output1": input_dict[self.input_variables[0].name] ** 2,
            "output2": input_dict[self.input_variables[1].name] ** 2,
        }
        return output_dict


m = ExampleModel("example_model.yml")
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
