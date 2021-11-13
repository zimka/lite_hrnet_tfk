from typing import List, Tuple

import numpy as np
import tensorflow as tf

from lite_hrnet_tfk.layers import ShuffleLayer, ChannelSplitLayer, ConvBlockLayer

# pylint: disable=too-many-ancestors

class _BaseModule(tf.keras.models.Model):
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


class StemModule(_BaseModule):
    """
    Input module in Lite-HRNet. Reduces HW 4 times.
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L164
    """
    def __init__(self, stem_channels=32, out_channels=32, name="stem"):
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


class ShuffleModule(_BaseModule):
    """
    Basic block in LiteNaiveHrModule.
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L327
    """
    def __init__(self, *, filters: int, strides: int, name: str):
        # TODO: branch1
        super().__init__(self, name=name)

        if strides != 1:
            raise NotImplementedError()
        self.split = ChannelSplitLayer(n_splits=2, name=f"{name}.split")
        # the following works only when in_channels == out_channels,
        # which should be always true if module is according to paper
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

    def call(self, inputs):
        x, y = self.split(inputs)
        for l in self.branch2:
            y = l(y)
        x = self.concat([x, y])
        x = self.shuffle(x)
        return x


class FusionModule(_BaseModule):
    def __init__(self, *, branches_chan_list: List[int], name: str):
        super().__init__(self, name=name)
        self.to_from = list(list() for _ in branches_chan_list)
        # branch channels are sorted from the highest scale to the lowest
        for i_dst, chan_dst in enumerate(branches_chan_list):
            for j_src, chan_src in enumerate(branches_chan_list):
                self.to_from[i_dst].append(self._fuse_layer(i_dst, j_src, chan_dst, chan_src))
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs: List[tf.Tensor]):
        x_list = inputs
        assert len(x_list) == len(self.to_from)

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
            # https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L547
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


class LiteNaiveHrModule(_BaseModule):
    def __init__(self, *, num_blocks, branches_chan_list, name):
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

    def call(self, x_list):
        assert len(x_list) == len(self.branches)
        y_list = []
        for branch_idx, x in enumerate(x_list):
            for m in self.branches[branch_idx]:
                x = m(x)
            y_list.append(x)
        y_list = self.fuse(y_list)
        return y_list


class TransitionModule(_BaseModule):
    def __init__(self, *, num_channels_list: List[int], name: str):
        super().__init__(self, name=name)
        self.num_channels_list = num_channels_list

    def build(self, input_shape: Tuple[int]):
        # in theory transition module could add more than one downscaled fork,
        # or not add any fork at all,
        # but in paper only N:(N+1) transitions are used, because
        # transitions are used only at the beginning of the stage and
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


class StageModule(_BaseModule):
    def __init__(self, *, num_modules: int, num_blocks: int, num_channels_list: List[int],
        with_transition: bool = True, name):
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
