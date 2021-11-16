import numpy as np
import pytest
import tensorflow as tf

from lite_hrnet_tfk.modules.lite import CrossResolutionWeightingModule, ConditionalChannelWeightingModule, LiteHrModule

@pytest.mark.parametrize("n_scales", [2, 4])
def test_cross_resolution(n_scales, channels=40, hw=64, batch=2):
    inputs = []
    input_channels = []
    for n in range(n_scales):
        hw_n = hw // 2 ** n
        ch_n = channels * 2 **n
        inp = np.arange(batch * hw_n * hw_n * ch_n).reshape((batch, hw_n, hw_n, ch_n)).astype(np.float32)

        input_channels.append(ch_n)
        inputs.append(tf.convert_to_tensor(inp))

    md = CrossResolutionWeightingModule(channels_list=input_channels)
    outputs = md(inputs)
    assert len(outputs) == n_scales
    for inp, out in zip(inputs, outputs):
        assert inp.shape == out.shape



@pytest.mark.parametrize("n_scales", [2, 4])
@pytest.mark.parametrize("ratios", [4, 8])
def test_cond_channel_weighting(n_scales, ratios, channels=40, hw=64, batch=2):
    inputs = []
    input_channels = []
    for n in range(n_scales):
        hw_n = hw // 2 ** n
        ch_n = channels * 2 **n
        inp = np.arange(batch * hw_n * hw_n * ch_n).reshape((batch, hw_n, hw_n, ch_n)).astype(np.float32)

        input_channels.append(ch_n)
        inputs.append(tf.convert_to_tensor(inp))

    md = ConditionalChannelWeightingModule(ratios)
    outputs = md(inputs)
    assert len(outputs) == n_scales
    for inp, out in zip(inputs, outputs):
        assert inp.shape == out.shape


@pytest.mark.parametrize("n_scales", [2, 4])
@pytest.mark.parametrize("ratios", [4, 8])
def test_lite_hr_module(n_scales, ratios, num_blocks=2, channels=40, hw=64, batch=2):
    inputs = []
    input_channels = []
    for n in range(n_scales):
        hw_n = hw // 2 ** n
        ch_n = channels * 2 **n
        inp = np.arange(batch * hw_n * hw_n * ch_n).reshape((batch, hw_n, hw_n, ch_n)).astype(np.float32)

        input_channels.append(ch_n)
        inputs.append(tf.convert_to_tensor(inp))

    md = LiteHrModule(num_blocks=num_blocks, reduce_ratio=ratios, branches_chan_list=input_channels)
    outputs = md(inputs)
    assert len(outputs) == n_scales
    for inp, out in zip(inputs, outputs):
        assert inp.shape == out.shape