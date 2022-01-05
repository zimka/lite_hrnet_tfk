# lite_hrnet_tfk

## Description
Unofficial `tensorflow.keras` implementation of Lite-HRNet ([Lite-HRNet: A Lightweight High-Resolution Network](https://arxiv.org/abs/2104.06403)).
Lite-HRNet has an official implementation: [official repo with mmpose configs](https://github.com/HRNet/Lite-HRNet). Lite-HRnet has also been merged into [mmpose](https://github.com/open-mmlab/mmpose).

This implementation is based on the official one, you can find commented links to respective parts here and there.

## Installation
The only dependency of this project is tensorflow.
You can clone the repo and install the package locally:
```
git clone https://github.com/zimka/lite_hrnet_tfk
pip install lite_hrnet_tfk/
```

Or you can install it from github directly:
```
pip install git+https://github.com/zimka/lite_hrnet_tfk.git

```
Because of ambiguity of `tensorflow` nor `tensorflow-gpu` packages, none is installed by default.

You can either install appropriate tensorflow in advance, or specify it as extra dependency for pip during the installation:
```
pip install lite_hrnet_tfk[tensorflow-gpu]
```

## Tests
If you want to run tests please install the package locally, then run `pytest`:
```
pytest lite_hrnet_tfk/tests

```

## How to use
The easiest way to build a network is to use prepared configs.
```python
from lite_hrnet_tfk.config import LiteHrnetConfig # config describes how the net should be built
from lite_hrnet_tfk.net import LiteHrnet # net uses config to compose separate modules into the network

config = LiteHrnetConfig.lite18(out_channels=42) # create prepared config with as many channels as necessary
print(config) # check config parameters, change whatever you want (or use other LiteHrnetConfig classmethod)
net = LiteHrnet(config)
```
You can also build net from separate modules, check `lite_hrnet_tfk.modules` code for details.
