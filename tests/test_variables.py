import pytest
from pydantic import ValidationError

from lume_model.variables import ScalarVariable, get_variable


class TestScalarVariable:
    def test_init(self):
        ScalarVariable(
            name="test",
            default_value=0.1,
            value_range=(0, 1),
            unit="m",
        )
        # missing name
        with pytest.raises(ValidationError):
            ScalarVariable(default_value=0.1, value_range=(0, 1))
        # constant variable
        ScalarVariable(
            name="test",
            default_value=0.5,
            value_range=None,
        )

    def test_validate_value(self):
        var = ScalarVariable(
            name="test",
            default_value=0.8,
            value_range=(0.0, 10.0),
            value_range_tolerance=1e-8,
        )
        var.validate_value(5.0)
        with pytest.raises(TypeError):
            var.validate_value(int(5))
        # test validation config
        var.validate_value(11.0, config="none")
        # range check with strictness flag
        with pytest.raises(ValueError):
            var.validate_value(11.0, config="error")
        # constant variable
        constant_var = ScalarVariable(
            name="test",
            default_value=1.3,
            is_constant=True,
            value_range_tolerance=1e-5,
        )
        constant_var.validate_value(1.3, config="error")
        with pytest.raises(ValueError):
            constant_var.validate_value(1.4, config="error")
        # test tolerance
        var.validate_value(10.0 + 1e-9, config="error")
        with pytest.raises(ValueError):
            var.validate_value(10.0 + 1e-7, config="error")
        constant_var.validate_value(1.3 + 1e-6, config="error")
        with pytest.raises(ValueError):
            constant_var.validate_value(1.3 + 1e-4, config="error")
        # test constant range validation
        with pytest.raises(ValueError):
            constant_var = ScalarVariable(
                name="test",
                default_value=1.3,
                is_constant=True,
                value_range_tolerance=1e-5,
                value_range=(1.3, 1.5),
            )


def test_get_variable():
    var = get_variable(ScalarVariable.__name__)(name="test")
    assert isinstance(var, ScalarVariable)


# @pytest.mark.parametrize(
#     "variable_name,default,value_range,axis_labels,x_min,y_min,x_max,y_max",
#     [
#         ("test", np.array([[1, 2,], [3, 4]]), [0, 1], ["x", "y"], 0, 0, 1, 1),
#         pytest.param(
#             "test", 1.0, [0, 1], ["x", "y"], 0, 0, 1, 1, marks=pytest.mark.xfail
#         ),
#         ("test", np.empty((3, 3)), [0, 1], ["x", "y"], 0, 0, 1, 1),
#     ],
# )
# def test_input_image_variable(
#     variable_name, default, value_range, axis_labels, x_min, y_min, x_max, y_max
# ):
#     # test correctly typed
#     ImageInputVariable(
#         name=variable_name,
#         default=default,
#         value_range=value_range,
#         axis_labels=axis_labels,
#         x_min=x_min,
#         y_min=y_min,
#         x_max=x_max,
#         y_max=y_max,
#     )
#
#     # test missing name
#     with pytest.raises(ValidationError):
#         ImageInputVariable(
#             default=default,
#             value_range=value_range,
#             axis_labels=axis_labels,
#             x_min=x_min,
#             y_min=y_min,
#             x_max=x_max,
#             y_max=y_max,
#         )
#
#     # test missing axis labels
#     with pytest.raises(ValidationError):
#         ImageInputVariable(
#             name=variable_name,
#             default=default,
#             value_range=value_range,
#             x_min=x_min,
#             y_min=y_min,
#             x_max=x_max,
#             y_max=y_max,
#         )


# @pytest.mark.parametrize(
#     "variable_name,default,axis_labels",
#     [
#         ("test", np.array([[1, 2,], [3, 4]]), ["x", "y"],),
#         pytest.param("test", 1.0, ["x", "y"], marks=pytest.mark.xfail),
#     ],
# )
# def test_output_image_variable(variable_name, default, axis_labels):
#     shape = default.shape
#     ImageOutputVariable(
#         name=variable_name, default=default, shape=shape, axis_labels=axis_labels,
#     )
#
#     # test missing name
#     with pytest.raises(ValidationError):
#         ImageOutputVariable(
#             default=default, shape=shape, axis_labels=axis_labels,
#         )
#
#     # test missing axis labels
#     with pytest.raises(ValidationError):
#         ImageOutputVariable(
#             name=variable_name, default=default,
#         )
#
#     # test missing value
#     ImageOutputVariable(
#         name=variable_name, axis_labels=axis_labels,
#     )


# @pytest.mark.parametrize(
#     "variable_name,default,value_range,axis_labels,x_min,y_min,x_max,y_max",
#     [
#         ("test", np.array([[1, 2,], [3, 4]]), [0, 1], ["x", "y"], 0, 0, 1, 1),
#         pytest.param(
#             "test", 1.0, [0, 1], ["x", "y"], 0, 0, 1, 1, marks=pytest.mark.xfail
#         ),
#     ],
# )
# def test_image_variable_shape(
#     variable_name, default, value_range, axis_labels, x_min, y_min, x_max, y_max
# ):
#     shape = default.shape
#
#     # test correctly typed
#     variable = ImageInputVariable(
#         name=variable_name,
#         default=default,
#         value_range=value_range,
#         axis_labels=axis_labels,
#         x_min=x_min,
#         y_min=y_min,
#         x_max=x_max,
#         y_max=y_max,
#     )
#
#     assert shape == variable.shape


# @pytest.mark.parametrize(
#     "variable_name,default,value_range,axis_labels,x_min,y_min,x_max,y_max",
#     [("test", np.array([[1, 2,], [3, 4]]), [0, 1], ["x", "y"], 0, 0, 1, 1)],
# )
# def test_input_image_variable_color_mode(
#     variable_name, default, value_range, axis_labels, x_min, y_min, x_max, y_max
# ):
#
#     random_rgb_default = np.random.rand(10, 10, 3)
#
#     # test correctly typed
#     variable = ImageInputVariable(
#         name=variable_name,
#         default=random_rgb_default,
#         value_range=value_range,
#         axis_labels=axis_labels,
#         x_min=x_min,
#         y_min=y_min,
#         x_max=x_max,
#         y_max=y_max,
#     )
#
#     with pytest.raises(ValueError):
#         random_rgb_default = np.random.rand(10, 10, 2)
#         # test correctly typed
#         variable = ImageInputVariable(
#             name=variable_name,
#             default=random_rgb_default,
#             value_range=value_range,
#             axis_labels=axis_labels,
#             x_min=x_min,
#             y_min=y_min,
#             x_max=x_max,
#             y_max=y_max,
#         )


# @pytest.mark.parametrize(
#     "variable_name,default,value_range,dim_labels",
#     [
#         ("test", np.array([[1, 2,], [3, 4]]), [0, 5], ["x, y"]),
#         pytest.param("test", [0, 1], [0, 5], ["x", "y"], marks=pytest.mark.xfail),
#     ],
# )
# def test_input_array_variable(variable_name, default, value_range, dim_labels):
#     # test correctly typed
#     ArrayInputVariable(
#         name=variable_name,
#         default=default,
#         value_range=value_range,
#         dim_labels=dim_labels,
#     )
#
#     # test missing name
#     with pytest.raises(ValidationError):
#         ArrayInputVariable(
#             default=default, value_range=value_range, dim_labels=dim_labels,
#         )
#
#     # test missing axis labels
#     ArrayInputVariable(
#         name=variable_name, default=default, value_range=value_range,
#     )


# @pytest.mark.parametrize(
#     "variable_name,default,dim_labels",
#     [
#         ("test", np.array([[1, 2,], [3, 4]]), ["x", "y"],),
#         pytest.param("test", 1.0, ["x", "y"], marks=pytest.mark.xfail),
#     ],
# )
# def test_output_array_variable(variable_name, default, dim_labels):
#     shape = default.shape
#     ArrayOutputVariable(
#         name=variable_name, default=default, shape=shape, dim_labels=dim_labels,
#     )
#
#     # test missing name
#     with pytest.raises(ValidationError):
#         ArrayOutputVariable(
#             default=default, shape=shape, dim_labels=dim_labels,
#         )
#
#     # test missing labels
#     ArrayOutputVariable(
#         name=variable_name, default=default,
#     )
#
#     # test missing value
#     ArrayOutputVariable(
#         name=variable_name, dim_labels=dim_labels,
#     )


# @pytest.mark.parametrize(
#     "rows,variables",
#     [
#         (
#             None,
#             {
#                 "col1": {
#                     "row1": ScalarInputVariable(
#                         name="col1_row1", default=0,value_range=[-1, -1]
#                     ),
#                     "row2": ScalarInputVariable(
#                         name="col1_row2", default=0,value_range=[-1, 1]
#                     ),
#                 },
#                 "col2": {
#                     "row1": ScalarInputVariable(
#                         name="col2_row1", default=0,value_range=[-1, -1]
#                     ),
#                     "row2": ScalarInputVariable(
#                         name="col2_row2", default=0,value_range=[-1, 1]
#                     ),
#                 },
#             },
#         ),
#         pytest.param(
#             ["row1", "row2"],
#             {
#                 "col1": {
#                     "row1": ScalarInputVariable(
#                         name="col1_row1", default=0, value_range=[-1, -1]
#                     ),
#                     "row2": 5,
#                 },
#                 "col2": {
#                     "row1": ScalarInputVariable(
#                         name="col2_row1", default=0,value_range=[-1, -1]
#                     ),
#                     "row2": ScalarInputVariable(
#                         name="col2_row2", default=0,value_range=[-1, 1]
#                     ),
#                 },
#             },
#             marks=pytest.mark.xfail,
#         ),
#         pytest.param(
#             None,
#             {
#                 "col1": ArrayInputVariable(
#                     name="test", default=np.array([1, 2]), value_range=[0, 10]
#                 ),
#                 "col2": {
#                     "row1": ScalarInputVariable(
#                         name="col2_row1", default=0, value_range=[-1, -1]
#                     ),
#                     "row2": ScalarInputVariable(
#                         name="col2_row2", default=0, value_range=[-1, 1]
#                     ),
#                 },
#             },
#             marks=pytest.mark.xfail,
#         ),
#         (
#             ["row1", "row2"],
#             {
#                 "col1": ArrayInputVariable(
#                     name="test", default=np.array([1, 2]), value_range=[0, 10]
#                 ),
#                 "col2": {
#                     "row1": ScalarInputVariable(
#                         name="col2_row1", default=0, value_range=[-1, -1]
#                     ),
#                     "row2": ScalarInputVariable(
#                         name="col2_row2", default=0, value_range=[-1, 1]
#                     ),
#                 },
#             },
#         ),
#         pytest.param(
#             ["row1", "row2"],
#             {
#                 "col1": ArrayInputVariable(
#                     name="test", default=np.array([1, 2, 3, 4]), value_range=[0, 10]
#                 ),
#                 "col2": {
#                     "row1": ScalarInputVariable(
#                         name="col2_row1", default=0, value_range=[-1, -1]
#                     ),
#                     "row2": ScalarInputVariable(
#                         name="col2_row2", default=0, value_range=[-1, 1]
#                     ),
#                 },
#             },
#             marks=pytest.mark.xfail,
#         ),
#     ],
# )
# def test_variable_table(rows, variables):
#     if rows:
#         table_var = TableVariable(table_rows=rows, table_data=variables)
#     else:
#         table_var = TableVariable(table_data=variables)
#
#     with pytest.raises(ValueError):
#         table_var = TableVariable(table_data=None)
