import numpy as np
import tensorflow as tf

from lite_hrnet_tfk.net import lite_hrnet, LiteHrnet
from lite_hrnet_tfk.config import LiteHrnetConfig


def test_func_api():
    config = LiteHrnetConfig.naive18()
    net = lite_hrnet(config=config)
    assert isinstance(net, tf.keras.models.Model)
    x = np.random.random((1, 256, 256, 3))
    y = net(x)


def test_subclass_api():
    config = LiteHrnetConfig.naive18()
    net = LiteHrnet(config=config)
    assert isinstance(net, tf.keras.models.Model)

    x = np.random.random((1, 256, 256, 3))
    y = net(x)
