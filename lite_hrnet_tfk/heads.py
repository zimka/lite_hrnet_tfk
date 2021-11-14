from typing import List
import tensorflow as tf

from lite_hrnet_tfk.layers import ConvBlockLayer
from lite_hrnet_tfk.modules import _BaseModule


class HrNetHeadV2(_BaseModule):
    def __init__(self, *, num_scales:int, out_channels: int, name: str):
        super().__init__(self, name=name)
        self.num_scales = num_scales
        self.out_channels = out_channels

        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=1,
            strides=1, use_bias=True, name=f"{name}.conv",
            padding='same'
        )
        self.ups = [lambda x: x]
        for n in range(1, num_scales):
            self.ups.append(
                tf.keras.layers.UpSampling2D(2**n, interpolation='bilinear', name=f"{name}.ups.{n}")
            )
        self.concat = tf.keras.layers.Concatenate(name=f"{name}.concat")

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        x_list = inputs
        assert len(x_list) == len(self.ups), (len(x_list), len(self.ups))
        y_list = []
        for x, up in zip(x_list, self.ups):
            y_list.append(up(x))
        y = self.concat(y_list)
        y = self.conv(y)
        return y


class HrNetHeadV1(_BaseModule):
    def __init__(self, *, scale_idx: int, out_channels: int, name: str):
        super().__init__(self, name=name)

        self.scale_idx = scale_idx
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=1,
            strides=1, use_bias=True, name=f"{name}.conv",
            padding='same'
        )

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        x = x_list[self.scale_idx]
        y = self.conv(x)
        return y


class IterativeHead(_BaseModule):
    """
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L272
    """
    def __init__(self, out_channels: int, name: str):
        super().__init__(name=name)
        self.out_channels = out_channels

    def build(self, input_shape):
        num_channels_list = [sh[-1] for sh in input_shape]
        self.projects = []
        for scale_idx, chan_dst in enumerate(num_channels_list):
            if scale_idx != 0:
                chan_dst = num_channels_list[scale_idx - 1]
            self.projects.append(
                self._build_projection(chan_dst, name=f"{self.name}.proj.{scale_idx}")
            )

        self.ups = []
        for n in range(len(input_shape) - 1):
            self.ups.append(
                tf.keras.layers.UpSampling2D(2, interpolation='bilinear', name=f"{self.name}.up{n}")
            )
        self.ups.append(lambda x: x)

        self.out = tf.keras.layers.Conv2D(
            filters=self.out_channels, kernel_size=1,
            strides=1, use_bias=True, name=f"{self.name}.out",
            padding='same'
        )
        super().build(input_shape)

    def _build_projection(self, chan_dst, name):
        """
        Separable convolution with intermediate batch norms
        """
        layers = []
        layers += ConvBlockLayer(filters=None, kernel_size=3, strides=1, name=f"{name}.0", relu=False).layers
        layers += ConvBlockLayer(filters=chan_dst, kernel_size=1, strides=1, name=f"{name}.1", relu=False).layers
        return tf.keras.models.Sequential(layers, name=name)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """
        x_list: [
            S0[H, W, C], S1[ H/ 2, W/2, 2*C], S2[H/4, W/4, 4*C], S3[H/8, W/8, 8*C]
        ]
        """
        x_list = inputs
        assert len(x_list) == len(self.ups) == len(self.projects)
        last_x = None
        for idx in range(len(x_list)-1, -1, -1):
            s = x_list[idx]
            if last_x is not None:
                last_x = self.ups[idx](last_x)
                s = s + last_x
            s = self.projects[idx](s)
            last_x = s
        y = self.out(last_x)
        return y
