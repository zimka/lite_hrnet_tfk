import numpy as np
import tensorflow as tf

from lite_hrnet_tfk.modules import StemModule, LiteNaiveHrModule, TransitionModule, StageModule
from lite_hrnet_tfk.heads import HrNetHeadV2, HrNetHeadV1, IterativeHead
from lite_hrnet_tfk.config import LiteHrnetConfig


class _LiteHrnetImpl:
    def __init__(self, *, config: LiteHrnetConfig):
        super().__init__()
        self._build_backbone(config)
        self._build_head(config)

    def _build_backbone(self, config):
        self.stem = StemModule(
            stem_channels=config.stem.stem_channels,
            out_channels=config.stem.out_channels,
            name=f"{config.name}.stem"
        )
        self.stages = []
        self.trans = []
        for spec_idx, spec in enumerate(config.stages):
            self.trans.append(TransitionModule(
                num_channels_list=spec.num_channels_list,
                name=f"{config.name}.trans.{spec_idx}"
            ))
            self.stages.append(StageModule(
                num_modules=spec.num_modules,
                num_blocks=spec.num_blocks,
                num_channels_list=spec.num_channels_list,
                name=f"{config.name}.stages.{spec_idx}"
            ))

    def _build_head(self, config):
        assert config.head.version in config.head._HEAD_VERSIONS
        outputs = None
        if config.head.version == 'v0':
            self.head = lambda x: x
        elif config.head.version == 'v1':
            self.head = HrNetHeadV1(
                scale_idx=config.head.v1_scale_idx,
                out_channels=config.head.out_channels,
                name=f"{config.name}.head_v1"
            )
        elif config.head.version == 'v2':
            self.head = HrNetHeadV2(
                num_scales=len(config.stages) + 1,
                out_channels=config.head.out_channels,
                name=f"{config.name}.head_v2"
            )
        elif config.head.version == 'vi':
            self.head = IterativeHead(
                out_channels=config.head.out_channels,
                name=f"{config.name}.head_vi"
            )
        else:
            raise NotImplementedError()

    def call(self, x):
        x = [self.stem(x)]
        assert len(self.stages) == len(self.trans)
        for tran, stage in zip(self.trans, self.stages):
            x = tran(x)
            x = stage(x)
        x = self.head(x)
        return x


def lite_hrnet(x=tf.keras.layers.Input((256, 256, 3)), *, config:LiteHrnetConfig):
    impl = _LiteHrnetImpl(config=config)
    y = impl.call(x)
    return tf.keras.models.Model(inputs=x, outputs=y)


class LiteHrnet(_LiteHrnetImpl, tf.keras.models.Model):
    def call(self, x):
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)
        return super().call(x)
