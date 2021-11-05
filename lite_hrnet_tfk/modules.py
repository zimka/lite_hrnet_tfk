import tensorflow as tf
from typing import List
from lite_hrnet_tfk.layers import ShuffleLayer, ChannelSplitLayer, ConvBlockLayer



class StemModule(tf.keras.models.Model):
    """
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L164
    """
    def __init__(self, stem_channels=32, out_channels=32, name="stem"):
        super().__init__(self, name=name)

        self.conv1 = ConvBlockLayer(filters=stem_channels, kernel_size=3, strides=2, name=f"{name}_conv1")
        self.split = ChannelSplitLayer(2, f"{name}_ChannelSplit")
        self.branch1 = [
            ConvBlockLayer(filters=None, kernel_size=3, strides=2, name=f"{name}_branch1_0"),
            ConvBlockLayer(filters=out_channels // 2, kernel_size=1, strides=1, name=f"{name}_branch1_1"),
        ]
        self.branch2 = [
            ConvBlockLayer(filters=stem_channels, kernel_size=1, strides=1, name=f"{name}_expand"),
            ConvBlockLayer(filters=None, kernel_size=3, strides=2, name=f"{name}_depthwise"),
            ConvBlockLayer(filters=out_channels // 2,   kernel_size=1, strides=1, name=f"{name}_linear")
        ]
        self.concat = tf.keras.layers.Concatenate()

    def call(self, x):
        x = self.conv1(x)
        x1, x2 = self.split(x)
        for l in self.branch1:
            x1 = l(x1)
        for l in self.branch2:
            x2 = l(x2)
        x = self.concat([x1, x2])
        return x


class ShuffleModule(tf.keras.models.Model):
    """
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L327
    """
    def __init__(self, *, filters, strides, name):
        # TODO: branch1
        super().__init__(self, name=name)

        if strides != 1:
            raise NotImplementedError()
        self.split = ChannelSplitLayer(n_splits=2, name=f"{name}.split")
        # the following works only when in_channels == out_channels,
        # which should be always true if module is according to paper
        self._filters = filters
        self.branch2 = [
            ConvBlockLayer(filters=filters // 2, kernel_size=1, strides=1, name=f"{name}br2.1"),
            ConvBlockLayer(filters=None, kernel_size=3, strides=strides, name=f"{name}.br2.2"),
            ConvBlockLayer(filters=filters // 2, kernel_size=1, strides=1, name=f"{name}.br2.3")
        ]
        self.concat = tf.keras.layers.Concatenate(name=f"{name}.concat")
        self.shuffle = ShuffleLayer(n_groups=2, name=f"{name}.shuffle")

    def build(self, input_shape):
        msg = f"Module input must have in_channels == out_channels, "\
        "got input_shape({input_shape}) with filters({self._filters}) in {self.name}"
        assert (input_shape[-1] == self._filters), msg
        return super().build(input_shape)

    def call(self, x):
        x1, x2 = self.split(x)
        for l in self.branch2:
            x2 = l(x2)
        x = self.concat([x1, x2])
        x = self.shuffle(x)
        return x


class FusionModule(tf.keras.models.Model):
    def __init__(self, *, branches_chan_list, name):
        super().__init__(self, name=name)
        self.fuse_layers = list(list() for _ in branches_chan_list)
        # branch channels are sorted from the highest scale to the lowest
        for i_dst, chan_dst in enumerate(branches_chan_list):
            for j_src, chan_src in enumerate(branches_chan_list):
                self.fuse_layers[i_dst].append(self._fuse_layer(i_dst, j_src, chan_dst, chan_src))
        self.relu = tf.keras.layers.ReLU()

    def call(self, x_list):
        assert len(x_list) == len(self.fuse_layers)

        y_list = []

        for i_dst, fuse_layers_dst in enumerate(self.fuse_layers):
            fused = None
            for j_src, x in enumerate(x_list):
                y = fuse_layers_dst[j_src](x)
                if fused is None:
                    fused = y
                else:
                    fused += y
            fused = self.relu(fused)
            y_list.append(fused)
        return y_list

    def _fuse_layer(self, i_dst, j_src, chan_dst, chan_src):
        if j_src == i_dst:
            return tf.keras.layers.Lambda(lambda x: x)
        name = f"{self.name}.{i_dst}-{j_src}"
        if j_src > i_dst:
            # source feature map is smaller than destination -> upsample source
            # https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L547
            return tf.keras.models.Sequential(
                ConvBlockLayer(
                    filters=chan_dst, kernel_size=1, strides=1,
                    name=f"{name}.c", relu=False
                ).layers +
                [tf.keras.layers.UpSampling2D(2**(j_src - i_dst), interpolation='nearest', name=f"{name}.u")],
                name=name
            )
        elif j_src < i_dst:
            # source feature map is bigger than destination -> downscale source
            conv_downsamples = []
            for k in range(i_dst - j_src - 1):
                conv_downsamples += ConvBlockLayer(filters=None, kernel_size=3, strides=2, name=f"{name}.c{k}_1_", relu=False).layers
                conv_downsamples += ConvBlockLayer(filters=chan_src, kernel_size=1, strides=1, name=f"{name}.c{k}_2_", relu=False).layers
            k = i_dst - j_src - 1
            conv_downsamples += ConvBlockLayer(filters=None, kernel_size=3, strides=2, name=f"{name}.c{k}_1_", relu=False).layers
            conv_downsamples += ConvBlockLayer(filters=chan_dst, kernel_size=1, strides=1, name=f"{name}.c{k}_2_", relu=True).layers
            return tf.keras.models.Sequential(conv_downsamples, name=name)


class LiteNaiveHrModule(tf.keras.models.Model):
    def __init__(self, *, num_blocks, branches_chan_list, name, with_fuse=True):
        super().__init__(self, name=name)

        self.branches_modules = []
        for branch_idx, branch_chan_dst in enumerate(branches_chan_list):
            self.branches_modules.append(
                self._make_one_branch(branch_chan_dst, num_blocks, name=f"{name}.b{branch_idx}")
            )
        self.fuse_module = None
        if with_fuse:
            self.fuse_module = FusionModule(branches_chan_list=branches_chan_list, name=f"{name}.fuse")

    def _make_one_branch(self, channels, num_blocks, strides=1, name='ShuffleModule'):
        # actually strides are always 1 here in https://github.com/HRNet/Lite-HRNet
        modules = [
            ShuffleModule(filters=channels, strides=strides, name=f"{name}.bl0")
        ]
        for b in range(1, num_blocks):
            modules.append(
                ShuffleModule(filters=channels, strides=strides, name=f"{name}.bl{b}")
            )
        return modules

    def call(self, x_list):
        assert len(x_list) == len(self.branches_modules)
        y_list = []
        for branch_idx, x in enumerate(x_list):
            for m in self.branches_modules[branch_idx]:
                x = m(x)
            y_list.append(x)
        if self.fuse_module is not None:
            y_list = self.fuse_module(y_list)
        return y_list


class TransitionModule(tf.keras.models.Model):
    def __init__(self, *, num_channels_list, name):
        super().__init__(self, name=name)
        self.num_channels_list = num_channels_list

    def build(self, input_shape):
        # technically transition module could add more than one downscaled fork,
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
            if is_fork or is_channel_different:
                strides = 2 if is_fork else 1
                layers = ConvBlockLayer(filters=None, kernel_size=3, strides=strides, name=f"{self.name}.{i_dst}.d", relu=False).layers
                layers += ConvBlockLayer(filters=chan_dst, kernel_size=1, strides=1, name=f"{self.name}.{i_dst}.p", relu=True).layers
                transitions.append(
                    tf.keras.models.Sequential(layers, name=f"{self.name}.{i_dst}")
                )
            else:
                transitions.append(lambda x: x)
        self.forwards = transitions[:-1]
        self.fork = transitions[-1]

    def call(self, x_list):
        y_list = []
        assert len(x_list) == len(self.forwards)
        for i, x in enumerate(x_list):
            y_list.append(
                self.forwards[i](x)
            )
        if self.fork:
            y_list.append(self.fork(x_list[-1]))
        return y_list


class StageModule(tf.keras.models.Model):
    def __init__(self, *, num_modules: int, num_blocks: int, num_channels_list: List[int], name):
        super().__init__(self, name=name)
        self.stage_modules = []
        for m_idx in range(num_modules):
            self.stage_modules.append(
                LiteNaiveHrModule(
                    num_blocks=num_blocks,
                    branches_chan_list=num_channels_list,
                    name=f"{name}.m{m_idx}"
                )
            )

    def call(self, x_list):
        for s in self.stage_modules:
            x_list = s(x_list)
        return x_list
