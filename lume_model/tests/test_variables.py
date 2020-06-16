import pytest
import numpy as np
from pydantic import ValidationError
from lume_model.variables import (
    ScalarInputVariable,
    ScalarOutputVariable,
    ImageInputVariable,
    ImageOutputVariable,
)


def test_dict_representations():

    # test that keys are
    pass


def test_input_image_variable():
    # test correctly typed
    var = ImageInputVariable(
        name="test", value=np.array([[1, 2,], [3, 4]]), value_range=[1, 2]
    )

    # test image value constraints
    with pytest.raises(ValidationError):
        ImageInputVariable(
            name="test", value=np.array([1, 2, 3, 4]), value_range=[1, 2]
        )

    # test missing name
    with pytest.raises(ValidationError):
        ImageInputVariable(value=np.array([[1, 2,], [3, 4]]), value_range=[1, 2])

    # test missing value
    with pytest.raises(ValidationError):
        ImageInputVariable(name="test", value_range=[1, 2])

    # test missing range
    with pytest.raises(ValidationError):
        ImageInputVariable(
            name="test", value=np.array([[1, 2,], [3, 4]]),
        )


def test_output_image_variable():
    # test correctly typed
    var = ImageOutputVariable(
        name="test", value=np.array([[1, 2,], [3, 4]]), value_range=[1, 2]
    )

    # test image value constraints
    with pytest.raises(ValidationError):
        ImageOutputVariable(
            name="test", value=np.array([1, 2, 3, 4]), value_range=[1, 2]
        )

    # test missing name
    with pytest.raises(ValidationError):
        ImageOutputVariable(value=np.array([[1, 2,], [3, 4]]), value_range=[1, 2])

    # test missing value
    ImageOutputVariable(name="test", value_range=[1, 2])

    # test missing range
    ImageOutputVariable(
        name="test", value=np.array([[1, 2,], [3, 4]]),
    )


def test_dataframe_construction():
    pass
