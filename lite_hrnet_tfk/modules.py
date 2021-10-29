import tensorflow as tf
from lite_hrnet_tfk.layers import ShuffleLayer, ChannelSplitLayer


class ConvModule:
    def __init__(self, *, filters, kernel_size, strides, name, relu=True):
        self.layers = []
        if filters is not None:
            assert kernel_size == 1
            self.layers.append(tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=False,
                name=f"{name}_DwConv"
            ))
        else:
            self.layers.append(tf.keras.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=False,
                name=f"{name}_Conv"
            ))
        self.layers.append(tf.keras.layers.BatchNormalization(name=f"{name}_BN"))
        if relu:
            self.layers.append(tf.keras.layers.ReLU(name=f"{name}_Relu"))

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x


class StemModule:
    """
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L164
    """
    def __init__(self, stem_channels=32, out_channels=32, name="stem"):
        self.conv1 = ConvModule(filters=stem_channels, kernel_size=3, strides=2, name=f"{name}_conv1")
        self.split = ChannelSplitLayer(2, f"{name}_ChannelSplit")
        self.branch1 = [
            ConvModule(filters=None, kernel_size=3, strides=2, name=f"{name}_branch1_0"),
            ConvModule(filters=out_channels // 2, kernel_size=1, strides=1, name=f"{name}_branch1_1"),
        ]
        self.branch2 = [
            ConvModule(filters=stem_channels,   kernel_size=1, strides=1, name=f"{name}_expand"),
            ConvModule(filters=None, kernel_size=3, strides=2, name=f"{name}_depthwise"),
            ConvModule(filters=out_channels // 2,   kernel_size=1, strides=1, name=f"{name}_linear")
        ]
        self.concat = tf.keras.layers.Concatenate()

    def __call__(self, x):
        x = self.conv1(x)
        x1, x2 = self.split(x)
        for l in self.branch1:
            x1 = l(x1)
        for l in self.branch2:
            x2 = l(x2)
        x = self.concat([x1, x2])
        return x


class ShuffleModule:
    """
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L327
    """
    def __init__(self, *, filters, strides, name):
        # TODO: branch1
        if strides != 1:
            raise NotImplementedError()
        self.split = ChannelSplitLayer(n_splits=2, name=f"{name}.split")
        # the following works only when in_channels == out_channels,
        # which should be always true if module is according to paper
        self.branch2 = [
            ConvModule(filters=filters // 2, kernel_size=1, strides=1, name=f"{name}br2.1"),
            ConvModule(filters=None, kernel_size=3, strides=strides, name=f"{name}.br2.2"),
            ConvModule(filters=filters // 2, kernel_size=1, strides=1, name=f"{name}.br2.3")
        ]
        self.concat = tf.keras.layers.Concatenate(name=f"{name}.concat")
        self.shuffle = ShuffleLayer(n_groups=2, name=f"{name}.shuffle")

    def __call__(self, x):
        x1, x2 = self.split(x)
        for l in self.branch2:
            x2 = l(x2)
        x = self.concat([x1, x2])
        x = self.shuffle(x)
        return x


class FusionModule:
    def __init__(self, *, branches_chan_list, name):
        self.name = name
        self.fuse_layers = list(list() for _ in branches_chan_list)
        # branch channels are sorted from the highest scale to the lowest
        for i_dst, chan_dst in enumerate(branches_chan_list):
            for j_src, chan_src in enumerate(branches_chan_list):
                self.fuse_layers[i_dst].append(self._fuse_layer(i_dst, j_src, chan_dst, chan_src))
        self.relu = tf.keras.layers.ReLU()

    def __call__(self, x_list):
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
            return lambda x: x
        name = f"{self.name}.{i_dst}-{j_src}."
        if j_src > i_dst:
            # source feature map is smaller than destination -> upsample source
            # https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L547
            return tf.keras.models.Sequential(
                ConvModule(
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
                conv_downsamples += ConvModule(filters=None, kernel_size=3, strides=2, name=f"{name}{k}_1_", relu=False).layers
                conv_downsamples += ConvModule(filters=chan_src, kernel_size=1, strides=1, name=f"{name}{k}_2_", relu=False).layers
            k = i_dst - j_src - 1
            conv_downsamples += ConvModule(filters=None, kernel_size=3, strides=2, name=f"{name}{k}_1_", relu=False).layers
            conv_downsamples += ConvModule(filters=chan_dst, kernel_size=1, strides=1, name=f"{name}{k}_2_", relu=True).layers
            return tf.keras.models.Sequential(conv_downsamples, name=name)


class LiteNaiveHrModule:
    def __init__(self, *, num_blocks, branches_chan_list, name, with_fuse=True):
        self.branches_modules = []
        for branch_idx, branch_chan_dst in enumerate(branches_chan_list):
            self.branches_modules.append(
                self._make_one_branch(branch_chan_dst, num_blocks, name=f"{name}.b{branch_idx}")
            )
        self.fuse_module = None
        if with_fuse:
            self.fuse_module = FusionModule(branches_chan_list=branches_chan_list, name=f"{name}.f")

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

    def __call__(self, x_list):
        assert len(x_list) == len(self.branches_modules)
        y_list = []
        for branch_idx, x in enumerate(x_list):
            for m in self.branches_modules[branch_idx]:
                x = m(x)
            y_list.append(x)
        if self.fuse_module is not None:
            y_list = self.fuse_module(y_list)
        return y_list


class TransitionModule:
    def __init__(self, *, branches_chan_list_dst, branches_chan_list_src, name):
        # technically transition module could add more than one downscaled fork,
        # or not add any fork at all,
        # but in paper only N:(N+1) transitions are used, because
        # every stage adds one and only one new branch.
        assert len(branches_chan_list_dst) - len(branches_chan_list_src) <= 1
        transitions = []

        for i_dst, chan_dst in enumerate(branches_chan_list_dst):
            is_fork = (i_dst == len(branches_chan_list_dst) - 1)
            chan_src = branches_chan_list_src[i_dst] if i_dst < len(branches_chan_list_src) else -1
            is_channel_different = chan_src != chan_dst
            if is_fork or is_channel_different:
                strides = 2 if is_fork else 1
                layers = ConvModule(filters=None, kernel_size=3, strides=strides, name=f"{name}.{i_dst}.d", relu=False).layers
                layers += ConvModule(filters=chan_dst, kernel_size=1, strides=1, name=f"{name}.{i_dst}.p", relu=True).layers
                transitions.append(
                    tf.keras.models.Sequential(layers, name=f"{name}.{i_dst}")
                )
            else:
                transitions.append(lambda x: x)
        self.forward_transitions = transitions[:-1]
        self.fork_downscale_transition = transitions[-1]

    def __call__(self, x_list):
        y_list = []
        assert len(x_list) == len(self.forward_transitions)
        for i, x in enumerate(x_list):
            y_list.append(
                self.forward_transitions[i](x)
            )
        if self.fork_downscale_transition:
            y_list.append(self.fork_downscale_transition(x_list[-1]))
        return y_list


class HrNetHeadV2:
    def __init__(self, *, num_scales:int, output_channels: int, name) :
        self.conv = tf.keras.layers.Conv2D(
            filters=output_channels, kernel_size=1,
            strides=1, use_bias=True, name=f"{name}.p",
            padding='same'
        )
        self.upsamplers = [lambda x: x]
        for n in range(1, num_scales + 1):
            self.upsamplers.append(
                tf.keras.layers.UpSampling2D(2**n, interpolation='bilinear')
            )
        self.concat = tf.keras.layers.Concatenate()

    def __call__(self, x_list):
        assert len(x_list) == len(self.upsamplers), (len(x_list), len(self.upsamplers))
        y_list = []
        for x, up in zip(x_list, self.upsamplers):
            y_list.append(up(x))
        y = self.concat(y_list)
        y = self.conv(y)
        return y



