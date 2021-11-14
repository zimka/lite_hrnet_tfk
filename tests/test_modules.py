import numpy as np
import pytest
import tensorflow as tf

from lite_hrnet_tfk.modules import StemModule, ShuffleModule, FusionModule, LiteNaiveHrModule, TransitionModule, StageModule


@pytest.mark.parametrize("stem_channels", [32, 64])
@pytest.mark.parametrize("out_channels", [32, 40])
def test_stem(stem_channels, out_channels, chans=3, hw=128, batch=2):
    md = StemModule(stem_channels=stem_channels, out_channels=out_channels)

    inp = np.arange(batch * hw * hw * chans).reshape((batch, hw, hw, chans)).astype(np.float32)
    inp_t = tf.convert_to_tensor(inp)
    out = md(inp_t).numpy()

    out_b, out_h, out_w, out_c = out.shape
    assert out_b == batch, "Batch mismatch"
    assert out_h == hw // 4, "H mismatch"
    assert out_w == hw // 4, "w mismatch"
    assert out_c == out_channels, "channels mismatch"


@pytest.mark.parametrize("channels", [32, 40])
def test_shuffle(channels, hw=32, batch=2):
    md = ShuffleModule(filters=channels)

    inp = np.arange(batch * hw * hw * channels).reshape((batch, hw, hw, channels)).astype(np.float32)
    inp_t = tf.convert_to_tensor(inp)
    out = md(inp_t).numpy()

    out_b, out_h, out_w, out_c = out.shape
    assert out_b == batch, "Batch mismatch"
    assert out_h == hw, "H mismatch"
    assert out_w == hw, "w mismatch"
    assert out_c == channels, "channels mismatch"

    md = ShuffleModule(filters=channels * 2)
    try:
        out = md(inp_t)
    except:
        pass
    else:
        assert False, "ShuffleModule must raise when input channels != out channels"


@pytest.mark.parametrize("n_scales", [2, 4])
def test_fusion(n_scales, channels=40, hw=64, batch=2):
    inputs = []
    input_channels = []
    for n in range(n_scales):
        hw_n = hw // 2 ** n
        ch_n = channels * 2 **n
        inp = np.arange(batch * hw_n * hw_n * ch_n).reshape((batch, hw_n, hw_n, ch_n)).astype(np.float32)

        input_channels.append(ch_n)
        inputs.append(tf.convert_to_tensor(inp))

    md = FusionModule(branches_chan_list=input_channels)
    outputs = md(inputs)
    assert len(outputs) == n_scales
    for inp, out in zip(inputs, outputs):
        assert inp.shape == out.shape


@pytest.mark.parametrize("n_scales", [2, 4])
def test_lite_naive_hr_module(n_scales, channels=40, hw=64, batch=2):
    inputs = []
    input_channels = []
    for n in range(n_scales):
        hw_n = hw // 2 ** n
        ch_n = channels * 2 **n
        inp = np.arange(batch * hw_n * hw_n * ch_n).reshape((batch, hw_n, hw_n, ch_n)).astype(np.float32)

        input_channels.append(ch_n)
        inputs.append(tf.convert_to_tensor(inp))

    md = LiteNaiveHrModule(num_blocks=2, branches_chan_list=input_channels)

    outputs = md(inputs)
    assert len(outputs) == n_scales
    for inp, out in zip(inputs, outputs):
        assert inp.shape == out.shape


def test_transition(input_channels=(32, 80), out_channels=(40, 80, 160), hw=64, batch=2):
    inputs = []
    for n, ch_n in enumerate(input_channels):
        hw_n = hw // 2 ** n
        inp = np.arange(batch * hw_n * hw_n * ch_n).reshape((batch, hw_n, hw_n, ch_n)).astype(np.float32)
        inputs.append(tf.convert_to_tensor(inp))
    md = TransitionModule(num_channels_list=out_channels)

    outputs = md(inputs)

    assert len(outputs) == len(out_channels)
    for n, (ch, out) in enumerate(zip(out_channels, outputs)):
        hw_n = hw // 2 ** n
        assert out.shape[1] == hw_n
        assert out.shape[2] == hw_n
        assert out.shape[-1] == ch


def test_stage(num_modules=1, num_blocks=2, num_channels_list=(40, 80, 160), hw=64, batch=2):
    md = StageModule(num_modules=num_modules, num_blocks=num_blocks, num_channels_list=num_channels_list)
    inputs = []
    for n, ch_n in enumerate(num_channels_list[:-1]):
        hw_n = hw // 2 ** n
        inp = np.arange(batch * hw_n * hw_n * ch_n).reshape((batch, hw_n, hw_n, ch_n)).astype(np.float32)
        inputs.append(tf.convert_to_tensor(inp))


    outputs = md(inputs)

    assert len(outputs) == len(num_channels_list)
    for n, (ch, out) in enumerate(zip(num_channels_list, outputs)):
        hw_n = hw // 2 ** n
        assert out.shape[1] == hw_n
        assert out.shape[2] == hw_n
        assert out.shape[-1] == ch
