# lume-model

Lume-model holds data structures used in the LUME modeling toolset. Variables and models built using lume-model will be compatible with other tools. Lume-model uses [pydantic](https://pydantic-docs.helpmanual.io/) models to enforce typed attributes upon instantiation.

## Requirements
* Python >= 3.7
* pydantic
* numpy

## Install

Lume-model can be installed with conda using the command:

``` $ conda install lume-model -c jrgarrahan ```

# Variables

The lume-model variables are intended to enforce requirements for input and output variables by variable type. Current variable implementations are scalar (float) or image (numpy array) type.

Example of minimal implementation of scalar input and output variables:
```python
input_variable = ScalarInputVariable(name="test_input", default=0.1, value_range=[1, 2])
output_variable = ScalarOutputVariable(name="test_output")
```

Example of minimal implementation of image input and output variables:
```python
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
    shape=(2,2),
    axis_labels=["count_1", "count_2"],
)
```

## Surrogate models

Lume-model model classes are intended to guide user development while allowing for flexibility and customizability. The base class `lume_model.models.SurrogateModel` is used to enforce LUME tool compatable classes for the execution of trained models. For this case, model loading and execution should be organized into class methods.

Surrogate Model Requirements:
* input_variables, output_variables: lume-model input and output variables are required for use with lume-epics tools. The user can optionally define these as class attributes or design the subclass so that these are passed during initialization . Names of all variables must be unique in order to be served using the EPICS tools. A utility function for saving these variables, which also enforces the uniqueness constraint, is provided (lume_model.utils.save_variables).
* evaluate: The evaluate method is called by the serving model. Subclasses must implement the method, accepting a list of input variables and returning a list of the model's output variables with value attributes updated based on model execution.

Example model implementation:

```python
class ExampleModel(SurrogateModel):
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
