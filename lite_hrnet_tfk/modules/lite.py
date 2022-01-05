"""
Lite (not-naive) version of LiteHRNetModule, which uses weighting.
"""
from typing import List, Tuple

import tensorflow as tf
import tensorflow.keras as tfk

from lite_hrnet_tfk.layers import ConvBlockLayer, SpatialWeightingLayer, ShuffleLayer, ChannelSplitLayer, AdaptiveAveragePooling2D
from .base import BaseModule, FusionModule


class CrossResolutionWeightingModule(BaseModule):
    """
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L52
    """
    def __init__(self, channels_list: List[int], reduce_ratio: int = 16, name: str = "CrossResolutionWeightingModule"):
        super().__init__(name)
        self.total_channels = sum(channels_list)
        self.pools = None
        self.concat = tfk.layers.Concatenate(name=f"{name}.concat")
        self.conv1 = tfk.layers.Conv2D(filters=int(self.total_channels/reduce_ratio), kernel_size=1, activation='relu', name=f"{name}.conv1")
        self.conv2 = tfk.layers.Conv2D(filters=self.total_channels, kernel_size=1, activation='sigmoid', name=f"{name}.conv2")
        self.split = None
        self.ups = None

    def build(self, input_shape:List[Tuple[int]]):
        minimal_hw = input_shape[-1][1:-1]
        self.pools = [
            AdaptiveAveragePooling2D(output_size=minimal_hw, name=f'{self.name}.pools.{num}')
            for num, _ in enumerate(input_shape)
        ]
        all_channels = [sh[-1] for sh in input_shape]
        self.split = ChannelSplitLayer(all_channels, name=f'{self.name}.split')
        self.ups = [
            tfk.layers.UpSampling2D((h//minimal_hw[0], w//minimal_hw[1]), name=f'{self.name}.ups.{num}')
            for num, (_, h, w, _) in enumerate(input_shape)
        ]
        super().build(input_shape)

    def call(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        x = [self.pools[i](y) for i, y in enumerate(inputs)]
        x = self.concat(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.split(x)
        return [up(y) for (up, y) in zip(self.ups, x)]


class ConditionalChannelWeightingModule(BaseModule):
    """
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L98
    """
    def __init__(self, reduce_ratio: int, strides: int = 1, name="ConditionalChannelWeightingModule"):
        super().__init__(name=name)
        if strides != 1:
            raise NotImplementedError("Only strides=1 is supported")
        self.reduce_ratio = reduce_ratio
        self.cross_resolution_weighting = None
        self.splits = None
        self.depthwise_convs = None
        self.spatial_weightings = None
        self.shuffles = None

    def build(self, input_shape: List[Tuple[int]]):
        channels_list = [sh[-1] for sh in input_shape]
        branch_channels_list = [c // 2 for c in channels_list]
        self.cross_resolution_weighting = CrossResolutionWeightingModule(
            channels_list=branch_channels_list,
            reduce_ratio=self.reduce_ratio,
            name=f"{self.name}.cross_resolution_weighting"
            )
        self.depthwise_convs = [
            ConvBlockLayer(
                filters=ch, kernel_size=3, strides=1, relu=False,
                name=f"{self.name}.depthwise_convs.{num}"
            )
            for num, ch in enumerate(branch_channels_list)
        ]
        self.spatial_weightings = [
            SpatialWeightingLayer(ratio=self.reduce_ratio, name=f"{self.name}.spatial_weightings.{num}")
            for num, ch in enumerate(branch_channels_list)
        ]
        self.splits = [
            ChannelSplitLayer(2, name=f"{self.name}.splits.{num}")
            for num, _ in enumerate(channels_list)
        ]
        self.concats = [
            tfk.layers.Concatenate(name=f"{self.name}.concats.{num}")
            for num, _ in enumerate(channels_list)
        ]
        self.shuffles = [
            ShuffleLayer(2)
            for _ in channels_list
        ]

    def call(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        x_list, y_list = [], []
        for i, inp in enumerate(inputs):
            x, y = self.splits[i](inp)
            x_list.append(x)
            y_list.append(y)


        y_list = self.cross_resolution_weighting(y_list)
        y_list = [dw(s) for s, dw in zip(y_list, self.depthwise_convs)]
        y_list = [sw(s) for s, sw in zip(y_list, self.spatial_weightings)]

        outputs = []
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            out = self.concats[i]([x, y])
            out = self.shuffles[i](out)
            outputs.append(out)
        return outputs


class LiteHrModule(BaseModule):
    """
    Lite module, which uses weighting.
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L442
    """
    def __init__(self, *, num_blocks: int, branches_chan_list: List[int], reduce_ratio: int = 8, name: str = "LiteHrModule"):
        super().__init__(self, name=name)

        self.blocks = [
            ConditionalChannelWeightingModule(reduce_ratio=reduce_ratio, name=f'{self.name}.blocks.{num}')
            for num in range(num_blocks)
        ]
        self.fuse = FusionModule(branches_chan_list=branches_chan_list, name=f"{name}.fuse")


    def call(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        x_list = inputs
        for block in self.blocks:
            x_list = block(x_list)
        x_list = self.fuse(x_list)
        return x_list