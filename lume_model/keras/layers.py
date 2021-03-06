"""
This module contains layers for use in building toolkit compatible models.
"""

import numpy as np
from tensorflow import keras
import logging

logger = logging.getLogger(__name__)


class LumeModelLayer(keras.layers.Layer):
    def _validate_args(self):

        for attr in ("_offset", "_scale", "_lower", "_upper"):
            try:
                val = getattr(self, attr)
            except:
                val = None

            if val:
                if not isinstance(val, (float, int,)):
                    raise ValueError(f"{attr} must be a float or int.")


class ScaleLayer(LumeModelLayer):
    """Layer for scaling float values.

    Attributes:
        _offset (float): Data offset
        _scale (float): Scale multiplier
        _lower (float): Lower range
        _upper (float): Upper range

    """

    trainable = False

    def __init__(
        self, offset: float, scale: float, lower: float, upper: float, **kwargs
    ) -> None:
        """Sets up scaling.

        Args:
            offset (float): Data offset
            scale (float): Scale multiplier
            lower (float): Lower range
            upper (float): Upper range

        """
        super(ScaleLayer, self).__init__(**kwargs)
        self._scale = scale
        self._offset = offset
        self._lower = lower
        self._upper = upper
        self._validate_args()

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Execute scaling on an array.

        Args:
            inputs (np.ndarray)

        Returns:
            np.ndarray
        """
        return self._lower + (
            (inputs - self._offset) * (self._upper - self._lower) / self._scale
        )

    def get_config(self) -> dict:
        """Get layer config.

        Returns:
            dict

        """
        return {
            "scale": self._scale,
            "offset": self._offset,
            "lower": self._lower,
            "upper": self._upper,
        }


class UnscaleLayer(LumeModelLayer):
    """Layer used for unscaling float values.

    Attributes:
        _offset (float): Data offset
        _scale (float): Scale multiplier
        _lower (float): Lower range
        _upper (float): Upper range

    """

    trainable = False

    def __init__(
        self, offset: float, scale: float, lower: float, upper: float, **kwargs
    ):
        """Sets up scaling.

        Args:
            offset (float): Data offset
            scale (float): Scale multiplier
            lower (float): Lower range
            upper (float): Upper range
        """
        super(UnscaleLayer, self).__init__(**kwargs)
        self._scale = scale
        self._offset = offset
        self._lower = lower
        self._upper = upper
        self._validate_args()

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Unscale an array

        Args:
            inputs (np.ndarray)

        """
        return (
            ((inputs - self._lower) * self._scale) / (self._upper - self._lower)
        ) + self._offset

    def get_config(self) -> dict:
        """Get layer config.

        Returns:
            dict

        """
        return {
            "scale": self.scale,
            "offset": self.offset,
            "lower": self.lower,
            "upper": self.upper,
        }


class UnscaleImgLayer(LumeModelLayer):
    """Layer used to unscale images.


    """

    trainable = False

    def __init__(self, offset: float, scale: float, **kwargs):
        """
        Args:
            offset (float): Data offset
            scale (float): Scale multiplier
        """
        super(UnscaleImgLayer, self).__init__(**kwargs)
        self._scale = scale
        self._offset = offset
        self._validate_args()

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Unscale an image.

        Returns:
            np.ndarray

        """
        return (inputs + self._offset) * self._scale

    def get_config(self) -> dict:
        """Get layer config.

        Returns:
            dict

        """
        return {"img_scale": self.img_scale, "img_offset": self.img_offset}
