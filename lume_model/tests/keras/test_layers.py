import pytest

# test value and failed initialization with characters
@pytest.mark.parametrize(
    "offset,scale,lower,upper",
    [
        (1, 2, 0, 1),
        (5, 4, -1, 1),
        pytest.param("t", "e", "s", "t", marks=pytest.mark.xfail),
    ],
)
@pytest.mark.keras_toolkit
def test_scale_layer(offset, scale, lower, upper):
    ScaleLayer = pytest.importorskip("lume_model.keras.layers.ScaleLayer")
    scale_layer = ScaleLayer(offset, scale, lower, upper)


# test value and failed initialization with characters
@pytest.mark.parametrize(
    "offset,scale,lower,upper",
    [
        (1, 2, 0, 1),
        (5, 4, -1, 1),
        pytest.param("t", "e", "s", "t", marks=pytest.mark.xfail),
    ],
)
@pytest.mark.keras_toolkit
def test_unscale_layer(offset, scale, lower, upper):
    UnscaleLayer = pytest.importorskip("lume_model.keras.layers.UnscaleLayer")
    unscale_layer = UnscaleLayer(offset, scale, lower, upper)


# test value and failed initialization with characters
@pytest.mark.parametrize(
    "offset,scale", [(1, 2), (5, 4), pytest.param("t", "e", marks=pytest.mark.xfail),],
)
@pytest.mark.keras_toolkit
def test_unscale_image_layer(offset, scale):
    UnscaleImgLayer = pytest.importorskip("lume_model.keras.layers.UnscaleImgLayer")
    unscale_layer = UnscaleImgLayer(offset, scale)
