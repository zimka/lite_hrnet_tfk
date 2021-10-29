import tensorflow as tf


class ShuffleLayer(tf.keras.layers.Layer):
    def __init__(self, n_groups: int = 2, name='Shuffle'):
        super().__init__(trainable=False, name=name)
        self.n_groups = n_groups
        self._shuffle_shape = None
        self._shuffle_perm = None

    def build(self, input_shape):
        channels_num = input_shape[-1]
        err_msg = f"Can't divide {channels_num} channels for n_groups={self.n_groups}"
        assert channels_num % self.n_groups == 0, err_msg

        self._shuffle_shape = list(input_shape)[:-1] + [self.n_groups, channels_num // self.n_groups]
        if self._shuffle_shape[0] is None:
            self._shuffle_shape[0] = -1

        self._shuffle_perm = list(range(len(self._shuffle_shape)))
        self._shuffle_perm[-2:] = self._shuffle_perm[-2:][::-1]

    def call(self, x):
        assert self._shuffle_shape is not None, "Shuffle layer was not built"

        y = tf.reshape(x, self._shuffle_shape, name=self.name + "_reshape")
        y = tf.transpose(y, self._shuffle_perm, name=self.name + "_transpose")
        return tf.reshape(y, tf.shape(x), name=self.name + "_reshape_back")

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_groups': self.n_groups,
        })
        return config

class ChannelSplitLayer(tf.keras.layers.Layer):
    def __init__(self, n_splits: int = 2, name='ChannelSplit'):
        super().__init__(trainable=False, name=name)
        self.n_splits = n_splits

    def build(self, input_shape):
        channels_num = input_shape[-1]
        err_msg = f"Can't divide {channels_num} channels for n_splits={self.n_splits}"
        assert channels_num % self.n_splits == 0, err_msg

    def call(self, x):
        return tf.split(x, self.n_splits, axis=-1, name=self.name + "_split")

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_splits': self.n_splits,
        })
        return config




