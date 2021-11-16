"""
Modules that are necessary for both lite and naive.
"""
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from lite_hrnet_tfk.layers import ChannelSplitLayer, ConvBlockLayer

# pylint: disable=too-many-ancestors

class BaseModule(tf.keras.models.Model):
    """
    Computes shape after .build() to show it in .summary().
    Inherited modules must call super().build()
    """
    def build(self, input_shape: Tuple[int]):
        super().build(input_shape)
        self._output_shape = self.compute_output_shape(input_shape)

    @property
    def output_shape(self):
        if not self.built:
            raise ValueError("Layer has not been built")
        return np.array(self._output_shape).tolist()

    def get_config(self):
        """
        Only functional(graph) models require .get_config
        """
        # pylint: disable=abstract-method
        pass


class FusionModule(BaseModule):
    """
    Fuses different scales in the end of Lite*HrModule.
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L533
    """
    def __init__(self, *, branches_chan_list: List[int], name: str = "FusionModule"):
        super().__init__(self, name=name)
        self.to_from = list(list() for _ in branches_chan_list)
        # branch channels are sorted from the highest scale to the lowest
        for i_dst, chan_dst in enumerate(branches_chan_list):
            for j_src, chan_src in enumerate(branches_chan_list):
                self.to_from[i_dst].append(self._fuse_layer(i_dst, j_src, chan_dst, chan_src))
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs: List[tf.Tensor]):
        x_list = inputs
        assert len(x_list) == len(self.to_from), f"Input len is {len(x_list)}, but have {len(self.to_from)} branches"

        y_list = []

        for i_dst in range(len(x_list)):
            fused = None
            for j_src, x in enumerate(x_list):
                y = self.to_from[i_dst][j_src](x)
                if fused is None:
                    fused = y
                else:
                    fused += y
            fused = self.relu(fused)
            y_list.append(fused)
        return y_list

    def _fuse_layer(self, i_dst, j_src, chan_dst, chan_src):
        name = f"{self.name}.to_from.{i_dst}.{j_src}"
        if j_src > i_dst:
            # source feature map is smaller than destination -> upsample source
            return tf.keras.models.Sequential(
                ConvBlockLayer(
                    filters=chan_dst, kernel_size=1, strides=1,
                    name=f"{name}.0", relu=False
                ).layers +
                [tf.keras.layers.UpSampling2D(2**(j_src - i_dst), interpolation='nearest', name=f"{name}.1")],
                name=name
            )
        elif j_src < i_dst:
            # source feature map is bigger than destination -> downscale source
            conv_downsamples = []
            for k in range(i_dst - j_src - 1):
                conv_downsamples += ConvBlockLayer(
                    filters=None, kernel_size=3, strides=2, name=f"{name}.{2*k}", relu=False).layers
                conv_downsamples += ConvBlockLayer(filters=chan_src, kernel_size=1, strides=1, name=f"{name}.{2*k + 1}", relu=False).layers
            k = i_dst - j_src - 1
            conv_downsamples += ConvBlockLayer(filters=None, kernel_size=3, strides=2, name=f"{name}.{2*k}", relu=False).layers
            conv_downsamples += ConvBlockLayer(filters=chan_dst, kernel_size=1, strides=1, name=f"{name}.{2*k + 1}", relu=True).layers
            return tf.keras.models.Sequential(conv_downsamples, name=name)
        else:
            return lambda x: x
