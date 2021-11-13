from typing import Tuple

import tensorflow as tf


class ShuffleLayer(tf.keras.layers.Layer):
    """
    Shuffles channels (last dim) in tensor.
    https://arxiv.org/pdf/1707.01083.pdf
    """
    def __init__(self, n_groups:int = 2, name:str = 'Shuffle'):
        super().__init__(trainable=False, name=name)
        self.n_groups = n_groups
        self._shuffle_shape = None
        self._shuffle_perm = None

    def build(self, input_shape: Tuple[int]):
        channels_num = input_shape[-1]
        err_msg = f"Can't divide {channels_num} channels for n_groups={self.n_groups}"
        assert channels_num % self.n_groups == 0, err_msg

        self._shuffle_shape = list(input_shape)[:-1] + [self.n_groups, channels_num // self.n_groups]
        if self._shuffle_shape[0] is None:
            self._shuffle_shape[0] = -1

        self._shuffle_perm = list(range(len(self._shuffle_shape)))
        self._shuffle_perm[-2:] = self._shuffle_perm[-2:][::-1]

    def call(self, inputs: tf.Tensor):
        assert self._shuffle_shape is not None, "Shuffle layer was not built"

        y = tf.reshape(inputs, self._shuffle_shape, name=self.name + "_reshape")
        y = tf.transpose(y, self._shuffle_perm, name=self.name + "_transpose")
        return tf.reshape(y, tf.shape(inputs), name=self.name + "_reshape_back")


class ChannelSplitLayer(tf.keras.layers.Layer):
    """
    Splits layer on N chunks of equal size.
    """
    def __init__(self, n_splits: int = 2, name: str = 'ChannelSplit'):
        super().__init__(trainable=False, name=name)
        self.n_splits = n_splits

    def build(self, input_shape):
        channels_num = input_shape[-1]
        err_msg = f"Can't divide {channels_num} channels for n_splits={self.n_splits}"
        assert channels_num % self.n_splits == 0, err_msg

    def call(self, inputs: tf.Tensor):
        # pylint: disable=no-value-for-parameter, redundant-keyword-arg
        return tf.split(inputs, self.n_splits, axis=-1, name=self.name + "_split")


class ConvBlockLayer(tf.keras.layers.Layer):
    """
    Common building block - conv, bn, relu.
    """
    def __init__(self, *, filters:int, kernel_size:int, strides:int, relu:bool = True, name:str = "ConvBlock"):
        super().__init__(name=name)

        self.layers = []
        if filters is not None:
            self.layers.append(tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=False,
                name=f"{name}_Conv"
            ))
        else:
            self.layers.append(tf.keras.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=False,
                name=f"{name}_DwConv"
            ))
        self.layers.append(tf.keras.layers.BatchNormalization(name=f"{name}_BN"))
        if relu:
            self.layers.append(tf.keras.layers.ReLU(name=f"{name}_Relu"))

    def call(self, inputs: tf.Tensor):
        x = inputs
        for l in self.layers:
            x = l(x)
        return x
