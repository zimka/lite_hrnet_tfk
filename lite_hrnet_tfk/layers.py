from addict import Dict
import tensorflow as tf


class _BaseLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = Dict()

    def get_config(self):
        config = super().get_config().copy()
        config.update(self.config.to_dict())
        return config

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.name})>"


class ShuffleLayer(_BaseLayer):
    def __init__(self, n_groups: int = 2, name='Shuffle'):
        super().__init__(trainable=False, name=name)
        self.config.n_groups = n_groups
        self._shuffle_shape = None
        self._shuffle_perm = None

    def build(self, input_shape):
        channels_num = input_shape[-1]
        err_msg = f"Can't divide {channels_num} channels for n_groups={self.config.n_groups}"
        assert channels_num % self.config.n_groups == 0, err_msg

        self._shuffle_shape = list(input_shape)[:-1] + [self.config.n_groups, channels_num // self.config.n_groups]
        if self._shuffle_shape[0] is None:
            self._shuffle_shape[0] = -1

        self._shuffle_perm = list(range(len(self._shuffle_shape)))
        self._shuffle_perm[-2:] = self._shuffle_perm[-2:][::-1]

    def call(self, x):
        assert self._shuffle_shape is not None, "Shuffle layer was not built"

        y = tf.reshape(x, self._shuffle_shape, name=self.name + "_reshape")
        y = tf.transpose(y, self._shuffle_perm, name=self.name + "_transpose")
        return tf.reshape(y, tf.shape(x), name=self.name + "_reshape_back")


class ChannelSplitLayer(_BaseLayer):
    def __init__(self, n_splits: int = 2, name='ChannelSplit'):
        super().__init__(trainable=False, name=name)
        self.config.n_splits = n_splits

    def build(self, input_shape):
        channels_num = input_shape[-1]
        err_msg = f"Can't divide {channels_num} channels for n_splits={self.config.n_splits}"
        assert channels_num % self.config.n_splits == 0, err_msg

    def call(self, x):
        return tf.split(x, self.config.n_splits, axis=-1, name=self.name + "_split")


class ConvBlockLayer(_BaseLayer):
    def __init__(self, *, filters, kernel_size, strides, name, relu=True):
        super().__init__(name=name)
        self.layers = []
        if filters is not None:
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

    def call(self, x):
        for l in self.layers:
            x = l(x)
        return x
