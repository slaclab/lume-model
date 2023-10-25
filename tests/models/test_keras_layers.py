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
def test_scale_layer(offset, scale, lower, upper):
    layers = pytest.importorskip("lume_model.models.keras_layers")
    scale_layer = layers.ScaleLayer(offset, scale, lower, upper)


# test value and failed initialization with characters
@pytest.mark.parametrize(
    "offset,scale,lower,upper",
    [
        (1, 2, 0, 1),
        (5, 4, -1, 1),
        pytest.param("t", "e", "s", "t", marks=pytest.mark.xfail),
    ],
)
def test_unscale_layer(offset, scale, lower, upper):
    layers = pytest.importorskip("lume_model.models.keras_layers")
    unscale_layer = layers.UnscaleLayer(offset, scale, lower, upper)


# test value and failed initialization with characters
@pytest.mark.parametrize(
    "offset,scale", [(1, 2), (5, 4), pytest.param("t", "e", marks=pytest.mark.xfail),],
)
def test_unscale_image_layer(offset, scale):
    layers = pytest.importorskip("lume_model.models.keras_layers")
    unscale_layer = layers.UnscaleImgLayer(offset, scale)
