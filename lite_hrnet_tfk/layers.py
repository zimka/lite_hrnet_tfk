"""
Basic layers that are necessary for Lite-HRNet.
Layers here mostly use tensorflow api and either don't have trainable weights at all
or use simple operations and have simple structure.
More complex blocks are defined in modules.
"""
from typing import Tuple, Union, Iterable

import tensorflow as tf
import tensorflow.keras as tfk


class ShuffleLayer(tfk.layers.Layer):
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
        super().build(input_shape)


    def call(self, inputs: tf.Tensor):
        assert self._shuffle_shape is not None, "Shuffle layer was not built"

        y = tf.reshape(inputs, self._shuffle_shape, name=self.name + "_reshape")
        y = tf.transpose(y, self._shuffle_perm, name=self.name + "_transpose")
        return tf.reshape(y, tf.shape(inputs), name=self.name + "_reshape_back")


class ChannelSplitLayer(tfk.layers.Layer):
    """
    Splits layer on n_splits_or_sizes chunks of equal size OR
    on len(n_splits_or_sizes)
    """
    def __init__(self, n_splits_or_sizes: Union[int, Tuple[int]] = 2, name: str = 'ChannelSplit'):
        super().__init__(trainable=False, name=name)
        self.n_splits_or_sizes = n_splits_or_sizes

    def build(self, input_shape: Tuple[int]):
        channels_num = input_shape[-1]
        err_msg = f"Can't divide {channels_num} channels for n_splits_or_sizes={self.n_splits_or_sizes}"
        if isinstance(self.n_splits_or_sizes, int):
            assert channels_num % self.n_splits_or_sizes == 0, err_msg
        else:
            assert channels_num == sum(self.n_splits_or_sizes), err_msg
        super().build(input_shape)

    def call(self, inputs: tf.Tensor):
        # pylint: disable=no-value-for-parameter, redundant-keyword-arg
        return tf.split(inputs, self.n_splits_or_sizes, axis=-1, name=self.name + "_split")


class SpatialWeightingLayer(tfk.layers.Layer):
    """
    https://github.com/HRNet/Lite-HRNet/blob/hrnet/models/backbones/litehrnet.py#L17
    """
    def __init__(self, ratio: int = 16, name: str = "SpatialWeighting"):
        super().__init__(name=name)
        self.ratio = ratio
        self.pool = None
        self.conv1 = None
        self.conv2 = None

    def build(self, input_shape: Tuple[int]):
        channels = input_shape[-1]
        self.pool = tfk.layers.GlobalAveragePooling2D(name=f"{self.name}.pool")
        self.conv1 = tfk.layers.Conv2D(filters=int(channels/self.ratio), kernel_size=1, activation='relu', name=f"{self.name}.conv1")
        self.conv2 = tfk.layers.Conv2D(filters=channels, kernel_size=1, activation='sigmoid', name=f"{self.name}.conv2")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        out = self.pool(inputs)
        out = tf.expand_dims(out, axis=1)
        out = tf.expand_dims(out, axis=1)
        out = self.conv1(out)
        out = self.conv2(out)
        return inputs * out


class ConvBlockLayer(tfk.layers.Layer):
    """
    Common building block - conv, bn, relu.
    """
    def __init__(self, *, filters:int, kernel_size:int, strides:int, relu:bool = True, use_bn: bool = True, name:str = "ConvBlock"):
        super().__init__(name=name)

        self.layers = []
        if filters is not None:
            self.layers.append(tfk.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=False,
                name=f"{name}_Conv"
            ))
        else:
            self.layers.append(tfk.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=False,
                name=f"{name}_DwConv"
            ))
        if use_bn:
            self.layers.append(tfk.layers.BatchNormalization(name=f"{name}_BN"))
        if relu:
            self.layers.append(tfk.layers.ReLU(name=f"{name}_Relu"))

    def call(self, inputs: tf.Tensor):
        x = inputs
        for l in self.layers:
            x = l(x)
        return x


class AdaptiveAveragePooling2D(tfk.layers.Layer):
    """
    Simplified version of tensorflow_addons AdaptiveAveragePooling implementation.
    https://github.com/tensorflow/addons/blob/v0.15.0/tensorflow_addons/layers/adaptive_pooling.py
    """
    def __init__(
        self,
        output_size: Union[int, Iterable[int]],
        data_format: str = 'channels_last',
        **kwargs,
    ):
        self.data_format = data_format
        self.reduce_function = tf.reduce_mean
        self.output_size = output_size
        super().__init__(**kwargs)

    def call(self, inputs, *args):
        h_bins = self.output_size[0]
        w_bins = self.output_size[1]
        if self.data_format == "channels_last":
            split_cols = tf.split(inputs, h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            split_rows = tf.split(split_cols, w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            out_vect = self.reduce_function(split_rows, axis=[2, 4])
        else:
            split_cols = tf.split(inputs, h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            out_vect = self.reduce_function(split_rows, axis=[3, 5])
        return out_vect

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    self.output_size[0],
                    self.output_size[1],
                    input_shape[3],
                ]
            )
        else:
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    input_shape[1],
                    self.output_size[0],
                    self.output_size[1],
                ]
            )

        return shape

    def get_config(self):
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}
