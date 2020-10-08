import pytest
import numpy as np
from pydantic import ValidationError
from lume_model.variables import (
    ScalarInputVariable,
    ScalarOutputVariable,
    ImageInputVariable,
    ImageOutputVariable,
)


def test_input_scalar_variable():
    # test correctly typed
    ScalarInputVariable(name="test", default=0.1, value_range=[1, 2])

    # test incorrect value type
    with pytest.raises(ValidationError):
        ScalarInputVariable(
            name="test", default=np.array([1, 2, 3, 4]), value_range=[1, 2]
        )

    # test missing name
    with pytest.raises(ValidationError):
        ScalarInputVariable(default=0.1, value_range=[1, 2])

    # test missing default
    with pytest.raises(ValidationError):
        ScalarInputVariable(name="test", value_range=[1, 2])

    # test missing range
    with pytest.raises(ValidationError):
        ScalarInputVariable(
            name="test", default=0.1,
        )


@pytest.mark.parametrize(
    "variable_name,default,value_range,is_constant,assign,assignment",
    [
        ("test", 0.1, [1, 2], False, True, 2.0),
        pytest.param("test", 0.1, [1, 2], True, True, 2.0, marks=pytest.mark.xfail),
    ],
)
def test_constant_input_scalar_variable(
    variable_name, default, value_range, is_constant, assign, assignment
):

    variable = ScalarInputVariable(
        name=variable_name,
        default=default,
        value_range=value_range,
        is_constant=is_constant,
    )

    # test assignment
    if assign:
        variable.default = assignment


def test_output_scalar_variable():
    # test correctly typed
    ScalarOutputVariable(name="test", default=0.1, value_range=[1, 2])

    # test incorrect value type
    with pytest.raises(ValidationError):
        ScalarOutputVariable(
            name="test", default=np.array([1, 2, 3, 4]), value_range=[1, 2]
        )

    # test missing name
    with pytest.raises(ValidationError):
        ScalarOutputVariable(default=0.1, value_range=[1, 2])

    # test missing value
    ScalarOutputVariable(name="test", value_range=[1, 2])

    # test missing range
    ScalarOutputVariable(
        name="test", default=0.1,
    )


def test_input_image_variable():
    # test correctly typed
    test_array = np.array([[1, 2,], [3, 4]])
    ImageInputVariable(
        name="test",
        default=test_array,
        value_range=[1, 10],
        axis_labels=["count_1", "count_2"],
        x_min=0,
        y_min=0,
        x_max=5,
        y_max=5,
    )

    # test image value constraints
    with pytest.raises(ValidationError):
        ImageInputVariable(
            name="test",
            default=np.array([1, 2, 3, 4]),
            value_range=[1, 10],
            axis_labels=["count_1", "count_2"],
            x_min=0,
            y_min=0,
            x_max=5,
            y_max=5,
        )

    # test missing name
    with pytest.raises(ValidationError):
        ImageInputVariable(
            default=test_array,
            value_range=[1, 10],
            axis_labels=["count_1", "count_2"],
            x_min=0,
            y_min=0,
            x_max=5,
            y_max=5,
        )

    # test missing value
    with pytest.raises(ValidationError):
        ImageInputVariable(
            name="test",
            value_range=[1, 10],
            axis_labels=["count_1", "count_2"],
            x_min=0,
            y_min=0,
            x_max=5,
            y_max=5,
        )

    # test missing range
    with pytest.raises(ValidationError):
        ImageInputVariable(
            name="test",
            default=np.array([[1, 2,], [3, 4]]),
            axis_labels=["count_1", "count_2"],
            x_min=0,
            y_min=0,
            x_max=5,
            y_max=5,
        )

    # test missing axis labels
    with pytest.raises(ValidationError):
        ImageInputVariable(
            name="test",
            default=np.array([[1, 2,], [3, 4]]),
            shape=test_array.shape,
            value_range=[1, 10],
            x_min=0,
            y_min=0,
            x_max=5,
            y_max=5,
        )


def test_output_image_variable():
    # test correctly typed
    test_array = np.array([[1, 2,], [3, 4]])

    ImageOutputVariable(
        name="test",
        default=test_array,
        shape=test_array.shape,
        axis_labels=["count_1", "count_2"],
    )

    # test dim of image value must = 2
    with pytest.raises(ValidationError):
        ImageOutputVariable(
            name="test",
            default=np.array([1, 2, 3, 4]),
            shape=test_array.shape,
            axis_labels=["count_1", "count_2"],
        )

    # test missing name
    with pytest.raises(ValidationError):
        ImageOutputVariable(
            default=test_array,
            shape=test_array.shape,
            axis_labels=["count_1", "count_2"],
        )

    # test missing axis labels
    with pytest.raises(ValidationError):
        ImageOutputVariable(
            name="test", default=test_array,
        )

    # test missing value
    ImageOutputVariable(
        name="test", axis_labels=["count_1", "count_2"],
    )

    ImageOutputVariable(
        name="test",
        value=np.array([[1, 2,], [3, 4]]),
        shape=test_array.shape,
        axis_labels=["count_1", "count_2"],
    )


if __name__ == "__main__":

    test_constant_input_scalar_variable()
