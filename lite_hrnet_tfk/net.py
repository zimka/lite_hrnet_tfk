import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from lite_hrnet_tfk.modules import StemModule, LiteNaiveHrModule, TransitionModule, StageModule
from lite_hrnet_tfk.heads import HrNetHeadV2, HrNetHeadV1, IterativeHead
from lite_hrnet_tfk.config import LiteHrnetConfig


class LiteHrnet(tfk.models.Model):
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
        for spec_idx, spec in enumerate(config.stages):
            self.stages.append(StageModule(
                num_modules=spec.num_modules,
                num_blocks=spec.num_blocks,
                num_channels_list=spec.num_channels_list,
                naive=spec.naive,
                name=f"{config.name}.stages.{spec_idx}"
            ))

    def _build_head(self, config: LiteHrnetConfig):
        assert config.head.version in config.head._HEAD_VERSIONS
        outputs = None
        if config.head.version == 'v0':
            self.head = lambda x: x
        elif config.head.version == 'v1':
            self.head = HrNetHeadV1(
                scale_idx=config.head.v1_scale_idx,
                out_channels=config.head.out_channels,
                name=f"{config.name}.head"
            )
        elif config.head.version == 'v2':
            self.head = HrNetHeadV2(
                num_scales=len(config.stages) + 1,
                out_channels=config.head.out_channels,
                name=f"{config.name}.head"
            )
        elif config.head.version == 'vi':
            self.head = IterativeHead(
                out_channels=config.head.out_channels,
                name=f"{config.name}.head"
            )
        else:
            raise NotImplementedError()

    def call(self, x):
        if isinstance(x, np.ndarray):
            x = tf.convert_to_tensor(x)
        x = [self.stem(x)]
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x
