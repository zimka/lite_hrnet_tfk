"""
Naive version of LiteHRNetModule, which uses not weighting, but channels shuffle.
"""
from typing import List, Tuple

import tensorflow as tf

from lite_hrnet_tfk.layers import ConvBlockLayer, ShuffleLayer, ChannelSplitLayer
from .base import BaseModule, FusionModule


class ShuffleModule(BaseModule):
    """
    Basic block in LiteNaiveHrModule.
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L327
    """
    def __init__(self, *, filters: int, strides: int = 1, name: str = "ShuffleModule"):
        # TODO: branch1
        super().__init__(self, name=name)

        if strides != 1:
            # all original lite-hrnet configs use stride=1 here
            raise NotImplementedError()
        self.split = ChannelSplitLayer(2, name=f"{name}.split")
        # the following works only when in_channels == out_channels,
        # which should be always true if module is used according to the paper
        self._filters = filters
        self.branch2 = [
            ConvBlockLayer(filters=filters // 2, kernel_size=1, strides=1, name=f"{name}branch2.0"),
            ConvBlockLayer(filters=None, kernel_size=3, strides=strides, name=f"{name}.branch2.1"),
            ConvBlockLayer(filters=filters // 2, kernel_size=1, strides=1, name=f"{name}.branch2.2")
        ]
        self.concat = tf.keras.layers.Concatenate(name=f"{name}.concat")
        self.shuffle = ShuffleLayer(n_groups=2, name=f"{name}.shuffle")

    def build(self, input_shape: Tuple[int]):
        msg = f"Module input must have in_channels == out_channels, "\
        "got input_shape({input_shape}) with filters({self._filters}) in {self.name}"
        assert (input_shape[-1] == self._filters), msg
        return super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x, y = self.split(inputs)
        for l in self.branch2:
            y = l(y)
        x = self.concat([x, y])
        x = self.shuffle(x)
        return x


class LiteNaiveHrModule(BaseModule):
    """
    Naive module, based on ShuffleModule.
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L524
    """
    def __init__(self, *, num_blocks: int, branches_chan_list: List[int], name: str = "LiteNaiveHrModule"):
        super().__init__(self, name=name)

        self.branches = []
        for branch_idx, branch_chan_dst in enumerate(branches_chan_list):
            self.branches.append(
                self._make_one_branch(branch_chan_dst, num_blocks, name=f"{name}.branches.{branch_idx}")
            )
        self.fuse = FusionModule(branches_chan_list=branches_chan_list, name=f"{name}.fuse")

    def _make_one_branch(self, channels, num_blocks, strides=1, name=''):
        # actually strides are always 1 here in https://github.com/HRNet/Lite-HRNet
        modules = [
            ShuffleModule(filters=channels, strides=strides, name=f"{name}.0")
        ]
        for i in range(1, num_blocks):
            modules.append(
                ShuffleModule(filters=channels, strides=strides, name=f"{name}.{i}")
            )
        return modules

    def call(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        x_list = inputs
        assert len(x_list) == len(self.branches)
        y_list = []
        for branch_idx, x in enumerate(x_list):
            for m in self.branches[branch_idx]:
                x = m(x)
            y_list.append(x)
        y_list = self.fuse(y_list)
        return y_list
