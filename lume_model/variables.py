import numpy as np
from enum import Enum
from typing import Any, List, Union, Optional, Generic, TypeVar
from pydantic import BaseModel, Field, validator
from pydantic.generics import GenericModel


class PropertyBaseModel(GenericModel):
    """
    Generic base class used for the Variables. This extends the pydantic GenericModel to serialize properties.


    Notes
    -----
    Workaround for serializing properties with pydantic until
    https://github.com/samuelcolvin/pydantic/issues/935
    is solved. This solution is referenced in the issue.
    """

    @classmethod
    def get_properties(cls):
        return [
            prop
            for prop in dir(cls)
            if isinstance(getattr(cls, prop), property)
            and prop not in ("__values__", "fields")
        ]

    def dict(
        self,
        *,
        include: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        exclude: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> "DictStrAny":
        attribs = super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        props = self.get_properties()
        # Include and exclude properties
        if include:
            props = [prop for prop in props if prop in include]
        if exclude:
            props = [prop for prop in props if prop not in exclude]

        # Update the attribute dict with the properties
        if props:
            attribs.update({prop: getattr(self, prop) for prop in props})

        return attribs


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
            raise TypeError("Numpy array required")
        return v


class Image(np.ndarray):
    """
    Custom type validator for image array.


    Notes
    -----
    This should be expanded to check for color images.


    #dw, dh
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> np.ndarray:
        # validate data...
        if not isinstance(v, np.ndarray):
            raise TypeError("Numpy array required")

        if not v.ndim == 2:
            raise ValueError(
                f"Image array must have dim=2. Provided array has {v.ndim} dimensions"
            )

        return v


# define generic value type
Value = TypeVar("Value")


class Variable(PropertyBaseModel, Generic[Value]):
    """
    Minimum requirements for a Variable

    Attributes
    ----------
    name: str
        Name of the variable

    default: Value, optional
        Default value assigned to the variable

    units: str, optional
        Units associated with the value

    precision: int
        Precision to use for the value

    """

    name: str = Field(...)  # name required
    default: Optional[Value]  # default optionally required

    class Config:
        allow_population_by_field_name = True  # do not use alias only-init


class InputVariable(Variable, Generic[Value]):
    """
    Base generic class for input variables.

    Attributes
    ----------
    value: Value
        Value assigned to variable

    value_range: list
        Acceptable range for value

    """

    value: Value = Field(...)  # value is required for input variables
    value_range: list = Field(..., alias="range")  # range required


class OutputVariable(Variable, Generic[Value]):
    """
    Base generic class for output variables. Value and range assignment are optional.

    Attributes
    ----------
    value: Value, optional
        Value  assigned to variable

    value_range: list, optional
        Acceptable range for value

    """

    value: Optional[Value]
    value_range: Optional[list] = Field(alias="range")


class NDVariableBase:
    """
    Holds properties associated with numpy array variables.

    Attributes
    ----------
    shape: tuple
        Shape of the numpy n-dimensional array

    """

    @property
    def shape(self) -> tuple:
        return self.value.shape


class ImageVariable(BaseModel):
    variable_type = "image"
    axis_labels: List[str]
    axis_units: List[str] = None
    precision: int = 8
    x_min: float = None
    x_max: float = None
    y_min: float = None
    y_max: float = None


class ScalarVariable:
    variable_type = "scalar"
    units: Optional[str]  # required for some output displays
    precision: int = 8


class ImageInputVariable(InputVariable[Image], NDVariableBase, ImageVariable):
    """
    Class composition of image input, and numpy array base class.

    """

    pass


class ImageOutputVariable(OutputVariable[Image], NDVariableBase, ImageVariable):
    """
    Class composition of image output, and numpy array base class.

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
    """

    pass
