"""
Modules for a single stage of Lite-HRNet.
"""
from typing import List, Tuple

import tensorflow as tf

from lite_hrnet_tfk.layers import ConvBlockLayer, ChannelSplitLayer
from .base import BaseModule
from .naive import LiteNaiveHrModule
from .lite import LiteHrModule


class StemModule(BaseModule):
    """
    Input module in Lite-HRNet. Reduces HW 4 times.
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L164
    """
    def __init__(self, stem_channels=32, out_channels=32, name="StemModule"):
        super().__init__(self, name=name)

        self.conv1 = ConvBlockLayer(filters=stem_channels, kernel_size=3, strides=2, name=f"{name}.conv1")
        self.split = ChannelSplitLayer(2, f"{name}.split")
        self.branch1 = [
            ConvBlockLayer(filters=None, kernel_size=3, strides=2, name=f"{name}.branch1.0"),
            ConvBlockLayer(filters=out_channels // 2, kernel_size=1, strides=1, name=f"{name}.branch1.1"),
        ]
        self.branch2 = [
            ConvBlockLayer(filters=stem_channels, kernel_size=1, strides=1, name=f"{name}.branch2.0"),
            ConvBlockLayer(filters=None, kernel_size=3, strides=2, name=f"{name}.branch2.1"),
            ConvBlockLayer(filters=out_channels // 2,   kernel_size=1, strides=1, name=f"{name}.branch2.2")
        ]
        self.concat = tf.keras.layers.Concatenate(name=f"{name}.concat")

    def call(self, x):
        x = self.conv1(x)
        x, y = self.split(x)
        for l in self.branch1:
            x = l(x)
        for l in self.branch2:
            y = l(y)
        return self.concat([x, y])


class TransitionModule(BaseModule):
    """
    Preprocess module which is used before each stage to
    create new "fork".
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L755
    """
    def __init__(self, *, num_channels_list: List[int], name: str = "TransitionModule"):
        super().__init__(self, name=name)
        self.num_channels_list = num_channels_list

    def build(self, input_shape: Tuple[int]):
        # in theory transition module could add more than one downscaled fork,
        # or not add any fork at all,
        # but in paper only N:(N+1) transitions are used, because
        # transitions are placed only at the beginning of each stage and
        # every stage adds one and only one new branch.
        num_channels_list_src = [sh[-1] for sh in input_shape]
        num_channels_list_dst = self.num_channels_list
        assert len(num_channels_list_dst) - len(num_channels_list_dst) <= 1
        transitions = []

        for i_dst, chan_dst in enumerate(num_channels_list_dst):
            is_fork = (i_dst == len(num_channels_list_dst) - 1)
            chan_src = num_channels_list_src[i_dst] if i_dst < len(num_channels_list_src) else -1
            is_channel_different = chan_src != chan_dst
            name = f"{self.name}"
            if is_fork:
                name += ".fork"
            else:
                name += f".forwards.{i_dst}"
            if is_fork or is_channel_different:
                strides = 2 if is_fork else 1
                layers = ConvBlockLayer(filters=None, kernel_size=3, strides=strides, name=f"{name}.0", relu=False).layers
                layers += ConvBlockLayer(filters=chan_dst, kernel_size=1, strides=1, name=f"{name}.1", relu=True).layers
                transitions.append(
                    tf.keras.models.Sequential(layers, name=f"{self.name}.{i_dst}")
                )
            else:
                transitions.append(lambda x: x)
        self.forwards = transitions[:-1]
        self.fork = transitions[-1]

        super().build(input_shape)

    def call(self, inputs: List[tf.Tensor]):
        x_list = inputs
        y_list = []
        assert len(x_list) == len(self.forwards)
        for i, x in enumerate(x_list):
            y_list.append(
                self.forwards[i](x)
            )
        if self.fork:
            y_list.append(self.fork(x_list[-1]))
        return y_list


class StageModule(BaseModule):
    """
    Single stage of Lite-HRNet.
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L823
    Optionally builds transition module in addition to LiteHR*Modules.
    """
    def __init__(self, *, num_modules: int, num_blocks: int, num_channels_list: List[int],
        with_transition: bool = True, name: str = "StageModule"):
        super().__init__(self, name=name)
        self.trans = None
        if with_transition:
            self.trans = TransitionModule(num_channels_list=num_channels_list, name=f"{name}.trans")
        self.lite_mods = []

        for m_idx in range(num_modules):
            self.lite_mods.append(
                LiteNaiveHrModule(
                    num_blocks=num_blocks,
                    branches_chan_list=num_channels_list,
                    name=f"{name}.lite_mods.{m_idx}"
                )
            )

    def call(self, inputs: List[tf.Tensor]):
        x_list = inputs
        if self.trans is not None:
            x_list = self.trans(x_list)

        for s in self.lite_mods:
            x_list = s(x_list)
        return x_list
