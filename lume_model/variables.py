"""
This module contains definitions of LUME-model variables for use with lume tools.
The variables are divided into input and outputs, each with different minimal requirements.
Initiating any variable without the minimum requirements will result in an error.

For now, only scalar variables (floats) are supported.
"""
import logging
from typing import Optional, Generic, TypeVar
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# define generic value type
Value = TypeVar("Value")


class Variable(BaseModel, Generic[Value]):
    """
    Minimum requirements for a variable.

    Attributes:
        name: Name of the variable.
        value: Value assigned to the variable.
        precision: Precision to use for the value.
    """
    name: str
    value: Optional[Value] = None
    precision: Optional[int] = None


class ScalarVariable(BaseModel):
    """
    Base class used for constructing a scalar variable.

    Attributes:
        variable_type: Indicates scalar variable.
        units: Units associated with scalar value.
        parent_variable: Variable for which this is an attribute.
    """
    variable_type: str = "scalar"
    units: Optional[str] = None  # required for some output displays
    parent_variable: str = (
        None  # indicates that this variable is an attribute of another
    )


class InputVariable(Variable, Generic[Value]):
    """
    Base class for input variables.

    Attributes:
        default: Default value assigned to the variable.
        is_constant: Indicates whether the variable is constant.
    """
    default: Value  # required default
    is_constant: bool = Field(False)


class OutputVariable(Variable, Generic[Value]):
    """
    Base class for output variables. Value and range assignment are optional.

    Attributes:
        default: Default value assigned to the variable.
        value_range: Acceptable range for value.
    """
    default: Optional[Value] = None
    value_range: Optional[list] = Field(None, alias="range")


class ScalarInputVariable(InputVariable[float], ScalarVariable):
    """
    Variable used for representing a scalar input. Scalar variables hold float values.
    Initialization requires name, default, and value_range.

    Attributes:
        value_range: Acceptable range for value.

    Example:
        ```
        variable = ScalarInputVariable(name="example_input", default=0.1, value_range=[0.0, 1.0])
        ```
    """
    value_range: list[float]


class ScalarOutputVariable(OutputVariable[float], ScalarVariable):
    """
    Variable used for representing a scalar output. Scalar variables hold float values.
    Initialization requires name.

    Example:
        ```
        variable = ScalarOutputVariable(name="example_output")
        ```
    """
    pass


# class NumpyNDArray(np.ndarray):
#     """
#     Custom type validator for numpy ndarray.
#     """
#
#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate
#
#     @classmethod
#     def validate(cls, v: Any) -> np.ndarray:
#         # validate data...
#
#         if isinstance(v, list):
#             # conver to array, keep order
#             v = np.ndarray(v, order="K")
#
#         if not isinstance(v, np.ndarray):
#             logger.exception("A numpy array is required for the value")
#             raise TypeError("Numpy array required")
#         return v
#
#     class Config:
#         json_encoders = {
#             np.ndarray: lambda v: v.tolist(),  # may lose some precision
#         }


# class Image(np.ndarray):
#     """
#     Custom type validator for image array.
#
#     """
#
#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate
#
#     @classmethod
#     def validate(cls, v: Any) -> np.ndarray:
#         # validate data...
#         if not isinstance(v, np.ndarray):
#             logger.exception("Image variable value must be a numpy array")
#             raise TypeError("Numpy array required")
#
#         if (not v.ndim == 2 and not v.ndim == 3) or (v.ndim == 3 and v.shape[2] != 3):
#             logger.exception("Array must have dim=2 or dim=3 to instantiate image")
#             raise ValueError(
#                 f"Image array must have dim=2 or dim=3. Provided array has {v.ndim} dimensions"
#             )
#
#         return v


# class NDVariableBase:
#     """
#     Holds properties associated with numpy array variables.
#
#     Attributes:
#         shape (tuple): Shape of the numpy n-dimensional array
#     """
#
#     @property
#     def shape(self) -> tuple:
#         if self.default is not None:
#             return self.default.shape
#         else:
#             return None


# class ImageVariable(BaseModel, NDVariableBase):
#     """
#     Base class used for constructing an image variable.
#
#     Attributes:
#         variable_type (str): Indicates image variable.
#
#         axis_labels (List[str]): Labels to use for rendering axes.
#
#         axis_units (Optional[List[str]]): Units to use for rendering axes labels.
#
#         x_min_variable (Optional[str]): Scalar variable associated with image minimum x.
#
#         x_max_variable (Optional[str]): Scalar variable associated with image maximum x.
#
#         y_min_variable (Optional[str]): Scalar variable associated with image minimum y.
#
#         y_max_variable (Optional[str]): Scalar variable associated with image maximum y.
#     """
#
#     variable_type: str = "image"
#     axis_labels: List[str]
#     axis_units: Optional[List[str]]
#     x_min_variable: Optional[str]
#     x_max_variable: Optional[str]
#     y_min_variable: Optional[str]
#     y_max_variable: Optional[str]


# class ArrayVariable(BaseModel, NDVariableBase):
#     """
#     Base class used for constructing an array variable.  Array variables can capture
#     strings by passing `variable_type="string"` during initialization. Otherwise, the
#     value will default to an array of floats.
#
#     Attributes:
#         variable_type (str): Indicates array variable.
#
#         dim_labels (Optional[List[str]]): Labels to use for rendering axes.
#
#         units (Optional[List[str]]): Units to use for rendering axes labels.
#
#         value_type (Literal["float", "string"]): Type of value held by array.
#
#     """
#
#     variable_type: str = "array"
#     units: Optional[List[str]]  # required for some output displays
#     dim_labels: Optional[List[str]]
#     value_type: Literal["float", "string"] = "float"


# class ImageInputVariable(InputVariable[Image], ImageVariable):
#     """
#     Variable used for representing an image input. Image variable values must be two or
#     three dimensional arrays (grayscale, color, respectively). Initialization requires
#     name, axis_labels, default, x_min, x_max, y_min, y_max.
#
#     Attributes:
#
#         name (str): Name of the variable.
#         default (Value):  Default value assigned to the variable.
#         precision (Optional[int]): Precision to use for the value.
#         value (Optional[Value]): Value assigned to variable
#         value_range (list): Acceptable range for value
#         variable_type (str): Indicates image variable.
#         axis_labels (List[str]): Labels to use for rendering axes.
#         axis_units (Optional[List[str]]): Units to use for rendering axes labels.
#         x_min (float): Minimum x value of image.
#         x_max (float): Maximum x value of image.
#         y_min (float): Minimum y value of image.
#         y_max (float): Maximum y value of image.
#         x_min_variable (Optional[str]): Scalar variable associated with image minimum x.
#         x_max_variable (Optional[str]): Scalar variable associated with image maximum x.
#         y_min_variable (Optional[str]): Scalar variable associated with image minimum y.
#         y_max_variable (Optional[str]): Scalar variable associated with image maximum y.
#
#
#     Example:
#         ```
#         variable = ImageInputVariable(
#             name="test",
#             default=np.array([[1,4], [5,2]]),
#             value_range=[1, 10],
#             axis_labels=["count_1", "count_2"],
#             x_min=0,
#             y_min=0,
#             x_max=5,
#             y_max=5,
#         )
#         ```
#
#     """
#
#     x_min: float
#     x_max: float
#     y_min: float
#     y_max: float


# class ImageOutputVariable(OutputVariable[Image], ImageVariable):
#     """
#     Variable used for representing an image output. Image variable values must be two or
#     three dimensional arrays (grayscale, color, respectively). Initialization requires
#     name and axis_labels.
#
#     Attributes:
#         name (str): Name of the variable.
#         default (Optional[Value]):  Default value assigned to the variable.
#         precision (Optional[int]): Precision to use for the value.
#         value (Optional[Value]): Value assigned to variable
#         value_range (Optional[list]): Acceptable range for value
#         variable_type (str): Indicates image variable.
#         axis_labels (List[str]): Labels to use for rendering axes.
#         axis_units (Optional[List[str]]): Units to use for rendering axes labels.
#         x_min (Optional[float]): Minimum x value of image.
#         x_max (Optional[float]): Maximum x value of image.
#         y_min (Optional[float]): Minimum y value of image.
#         y_max (Optional[float]): Maximum y value of image.
#         x_min_variable (Optional[str]): Scalar variable associated with image minimum x.
#         x_max_variable (Optional[str]): Scalar variable associated with image maximum x.
#         y_min_variable (Optional[str]): Scalar variable associated with image minimum y.
#         y_max_variable (Optional[str]): Scalar variable associated with image maximum y.
#
#     Example:
#         ```
#         variable =  ImageOutputVariable(
#             name="test",
#             default=np.array([[2 , 1], [1, 4]]),
#             axis_labels=["count_1", "count_2"],
#         )
#
#         ```
#
#
#     """
#
#     x_min: Optional[float] = None
#     x_max: Optional[float] = None
#     y_min: Optional[float] = None
#     y_max: Optional[float] = None


# class ArrayInputVariable(InputVariable[NumpyNDArray], ArrayVariable):
#     """
#     Variable used for representing an array input. Array variables can capture
#     strings by passing `variable_type="string"` during initialization. Otherwise, the
#     value will default to an array of floats.
#
#     Attributes:
#         name (str): Name of the variable.
#         default (np.ndarray):  Default value assigned to the variable.
#         precision (Optional[int]): Precision to use for the value.
#         value (Optional[Value]): Value assigned to variable
#         value_range (Optional[list]): Acceptable range for value
#         variable_type (str): Indicates array variable.
#         dim_labels (List[str]): Labels to use for dimensions
#         dim_units (Optional[List[str]]): Units to use for dimensions.
#
#     """
#
#     pass


# class ArrayOutputVariable(OutputVariable[NumpyNDArray], ArrayVariable):
#     """
#     Variable used for representing an array output. Array variables can capture
#     strings by passing `variable_type="string"` during initialization. Otherwise, the
#     value will default to an array of floats.
#
#     Attributes:
#         name (str): Name of the variable.
#
#         default (Optional[np.ndarray]):  Default value assigned to the variable.
#
#         precision (Optional[int]): Precision to use for the value.
#
#         value (Optional[Value]): Value assigned to variable
#
#         value_range (Optional[list]): Acceptable range for value
#
#         variable_type (str): Indicates array variable.
#
#         dim_labels (List[str]): Labels to use for dimensions
#
#         dim_units (Optional[List[str]]): Units to use for dimensions.
#     """
#
#     pass


# class TableVariable(BaseModel):
#     """Table variables are used for creating tabular representations of data. Table variables should only be used for client tools.
#
#     Attributes:
#         table_rows (Optional[List[str]]): List of rows to assign to array data.
#         table_data (dict): Dictionary representation of columns and rows.
#         rows (list): List of rows.
#         columns (list): List of columns.
#     """
#
#     table_rows: Optional[List[str]] = None
#     table_data: dict
#
#     @property
#     def columns(self) -> tuple:
#         if self.table_data is not None:
#             return list(self.table_data.keys())
#         else:
#             return None
#
#     @validator("table_rows")
#     def validate_rows(cls, v):
#         if isinstance(v, list):
#             for val in v:
#                 if not isinstance(val, str):
#                     raise TypeError("Rows must be defined as strings")
#
#         else:
#             raise TypeError("Rows must be passed as list")
#
#         return v
#
#     @validator("table_data")
#     def table_data_formatted(cls, v, values) -> dict:
#         passed_rows = values.get("table_rows", None)
#         # validate data...
#         if not isinstance(v, dict):
#             logger.exception(
#                 "Must provide dictionary representation of table structure, outer level columns, inner level rows."
#             )
#             raise TypeError("Dictionary required")
#
#         # check that rows are represented in structure
#         for val in v.values():
#             if not isinstance(val, (dict, ArrayVariable)):
#                 logger.exception(
#                     "Rows are not represented in structure. Structure should map column title to either dictionary of row names and values or array variables."
#                 )
#                 raise TypeError(
#                     "Rows are not represented in structure. Structure should map column title to either dictionary of row names and values or array variables."
#                 )
#
#             if isinstance(val, ArrayVariable):
#                 if passed_rows is None:
#                     logger.exception("Must pass table_rows when using array variables.")
#                     raise TypeError("Must pass table_rows when using array variables.")
#
#                 # shape must match length of passed rows
#                 elif val.shape[0] != len(passed_rows):
#                     raise TypeError(
#                         "Array first dimension must match passed rows length."
#                     )
#
#         # check row structures to make sure properly formatted
#         for val in v.values():
#
#             # check row dictionary
#             if isinstance(val, dict):
#                 if val.get("variable_type", None) is None:
#                     for row_val in val.values():
#                         if not isinstance(row_val, (dict, ScalarVariable, float)):
#                             logger.exception(
#                                 "Row dictionary must map row names to ScalarVariables or float."
#                             )
#                             raise TypeError(
#                                 "Row dictionary must map row names to ScalarVariables or float."
#                             )
#
#                         # check that row keys align
#                         if isinstance(row_val, dict) and passed_rows is not None:
#                             row_rep = row_val.keys()
#                             for row in row_rep:
#                                 if row not in passed_rows:
#                                     raise TypeError(
#                                         f"Row {row} not in row list passed during construction."
#                                     )
#         return v
#
#     @property
#     def rows(self) -> tuple:
#         if self.table_rows is not None:
#             return self.table_rows
#         else:
#             struct_rows = []
#             for col, row_item in self.table_data.items():
#                 if isinstance(row_item, dict):
#                     struct_rows += list(row_item.keys())
#             return list(set(struct_rows))
