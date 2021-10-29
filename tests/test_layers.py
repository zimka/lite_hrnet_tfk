import numpy as np
import pytest
import tensorflow as tf

import lite_hrnet_tfk.layers as layers


def _functional_model_call(x, layer):
    b, h, w, c = x.shape

    inp_t = tf.keras.layers.Input((h, w, c))
    out_t = layer(inp_t)

    model = tf.keras.models.Model(
        inputs=inp_t,
        outputs=out_t
    )
    return model(x)


def _subclass_model_call(x, layer):
    class TestModel(tf.keras.models.Model):
        def __init__(self):
            super().__init__()
            self.layer = layer

        def call(self, inputs):
            return self.layer(inp_t)

    inp_t = tf.convert_to_tensor(x)
    model = TestModel()
    return model(inp_t)


@pytest.mark.parametrize("call", [_functional_model_call, _subclass_model_call])
@pytest.mark.parametrize("n_groups", [2, 6])
def test_shuffle(call, n_groups, chans=24, hw=5, batch=2):
    layer = layers.ShuffleLayer(n_groups=n_groups)

    inp = np.arange(batch * hw * hw * chans).reshape((batch, hw, hw, chans)).astype(np.float32)

    out = call(inp, layer).numpy()
    chan_idx_after = lambda c: n_groups * c % chans + n_groups * c // chans

    for b in range(batch):
        for c in range(chans):
            ca = chan_idx_after(c)
            before = inp[b, :, :, c]
            after = out[b, :, :, ca]
            assert np.allclose(before, after), (c, ca, before, after)


@pytest.mark.parametrize("call", [_functional_model_call, _subclass_model_call])
@pytest.mark.parametrize("n_splits", [2, 6])
def test_channel_split(call, n_splits, chans=24, hw=5, batch=2):
    layer = layers.ChannelSplitLayer(n_splits=n_splits)

    inp = np.arange(batch * hw * hw * chans).reshape((batch, hw, hw, chans)).astype(np.float32)

    out_t = call(inp, layer)
    assert len(out_t) == n_splits
    outs = [o.numpy() for o in out_t]
    inps = np.split(inp, n_splits, axis=-1)
    assert len(inps) == len(outs)

    for i, o in zip(inps, outs):
        assert i.shape == o.shape
        assert np.allclose(i, o), (i, o)

