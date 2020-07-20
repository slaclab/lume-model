"""
This module contains lume-model variable definitions.
"""

import numpy as np
from enum import Enum
import logging
from typing import Any, List, Union, Optional, Generic, TypeVar
from pydantic import BaseModel, Field, validator
from pydantic.generics import GenericModel

logger = logging.getLogger(__name__)


class NumpyNDArray(np.ndarray):
    """
    Custom type validator for numpy ndarray.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> np.ndarray:
        # validate data...
        if not isinstance(v, np.ndarray):
            logger.exception("A numpy array is required for the value")
            raise TypeError("Numpy array required")
        return v


class Image(np.ndarray):
    """
    Custom type validator for image array.

    Note:
        This should be expanded to check for color images.

    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> np.ndarray:
        # validate data...
        if not isinstance(v, np.ndarray):
            logger.exception("Image variable value must be a numpy array")
            raise TypeError("Numpy array required")

        if not v.ndim == 2:
            logger.exception("Array must have dim=2 to instantiate image")
            raise ValueError(
                f"Image array must have dim=2. Provided array has {v.ndim} dimensions"
            )

        return v


# define generic value type
Value = TypeVar("Value")


class Variable(BaseModel, Generic[Value]):
    """
    Minimum requirements for a Variable

    Attributes:
        name (str): Name of the variable.

        default (Value, optional):  Default value assigned to the variable

        precision (int, optional): Precision to use for the value

    """

    name: str = Field(...)  # name required
    default: Optional[Value]  # default optionally required
    precision: Optional[int] = 8

    class Config:
        allow_population_by_field_name = True  # do not use alias only-init


class InputVariable(Variable, Generic[Value]):
    """
    Base generic class for input variables.

    Attributes:
        name (str): Name of the variable.

        default (Value, optional):  Default value assigned to the variable

        precision (int, optional): Precision to use for the value

        value (Value): Value assigned to variable

        value_range (list): Acceptable range for value

    """

    value: Value = Field(...)  # value is required for input variables
    value_range: list = Field(..., alias="range")  # range required


class OutputVariable(Variable, Generic[Value]):
    """
    Base generic class for output variables. Value and range assignment are optional.

    Attributes:
        name (str): Name of the variable.

        default (Value, optional):  Default value assigned to the variable

        precision (int, optional): Precision to use for the value

        value (Value, optional): Value assigned to variable

        value_range (list, optional): Acceptable range for value

    """

    value: Optional[Value]
    value_range: Optional[list] = Field(alias="range")


class ImageVariable(BaseModel):
    """
    Base class used for constructing an image variable.

    Attributes:
        variable_type (tuple="image): Indicates image variable

        axis_labels (List[str]): Labels to use for rendering axes

        axis_units (Lsit[str]): Units to use for rendering axes labels

        x_min (float): Minimum x value of image

        x_max (float): Maximum x value of image

        y_min (float): Minimum y value of image

        y_max (float): Maximum y value of image
    """

    variable_type = "image"
    axis_labels: List[str]
    shape: tuple
    axis_units: List[str] = None
    x_min: float = None
    x_max: float = None
    y_min: float = None
    y_max: float = None


class ScalarVariable:
    """
    Base class used for constructing a scalar variable.

    Attributes:
        variable_type (tuple="scalar"): Type of variable

        units (str): Units associated with scalar value
    """

    variable_type = "scalar"
    units: Optional[str]  # required for some output displays


class ImageInputVariable(InputVariable[Image], ImageVariable):
    """
    Class composition of image input, and numpy array base class.

    Attributes:

    """

    pass


class ImageOutputVariable(OutputVariable[Image], ImageVariable):
    """
    Class composition of image output, and numpy array base class.

    Attributes:


    """

    pass


class ScalarInputVariable(InputVariable[float], ScalarVariable):
    """
    Class composition of scalar input and scalar base.
    """

    pass


class ScalarOutputVariable(OutputVariable[float], ScalarVariable):
    """
    Class composition of scalar output and scalar base.

    Attributes:

    """

    pass
