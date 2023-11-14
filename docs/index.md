# LUME-model

LUME-model holds data structures used in the LUME modeling toolset. Variables and models built using LUME-model will be compatible with other tools. LUME-model uses [Pydantic](https://pydantic-docs.helpmanual.io/) models to enforce typed attributes upon assignment.

## Installing LUME-model

LUME-model can be installed with conda using the command:
```shell
conda install lume-model -c conda-forge
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

Models and variables can be saved and loaded from YAML files, e.g. `m.dump("example_model.yml")` writes the following to file

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

## Developer

Clone this repository:
```shell
git clone https://github.com/slaclab/lume-model.git
```

Create an environment lume-model-dev with all the dependencies:
```shell
conda env create -f dev-environment.yml
```

Install as editable:
```shell
conda activate lume-model-dev
pip install --no-dependencies -e .
```
