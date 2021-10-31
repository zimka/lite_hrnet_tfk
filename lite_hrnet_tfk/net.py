import tensorflow as tf

from lite_hrnet_tfk.modules import StemModule, LiteNaiveHrModule, TransitionModule, HrNetHeadV2
from lite_hrnet_tfk.config import LiteHrnetConfig


def lite_hr_net(x=tf.keras.layers.Input((256, 256, 3)), *, config:LiteHrnetConfig, return_model=True):
    stem_m = StemModule(
        stem_channels=config.stem.stem_channels,
        out_channels=config.stem.out_channels,
        name=f"{config.name}.stem"
    )
    inputs = x
    x = stem_m(x)
    x = [x]
    branches_chan_list_before = [config.stem.out_channels]
    branches_chan_list_after = []
    for spec_idx, spec in enumerate(config.stages):
        branches_chan_list_after = spec.num_channels_list
        trans_m = TransitionModule(
            branches_chan_list_dst=branches_chan_list_after,
            branches_chan_list_src=branches_chan_list_before,
            name=f"{config.name}.s{spec_idx}.t"
        )
        x = trans_m(x)
        for m_idx in range(spec.num_modules):
            lhr_m = LiteNaiveHrModule(
                num_blocks=spec.num_blocks,
                branches_chan_list=branches_chan_list_after,
                name=f"{config.name}.s{spec_idx}.m{m_idx}")
            x = lhr_m(x)
        branches_chan_list_before = branches_chan_list_after

    assert config.head.version in config.head._HEAD_VERSIONS
    outputs = None
    if config.head.version == 'v0':
        outputs = x
    elif config.head.version == 'v1':
        outputs = x[config.head.v1_scale_idx]
    elif config.head.version == 'v2':
        head = HrNetHeadV2(num_scales=len(config.stages), output_channels=config.head.v2_out_channels, name=f"{config.name}.head")
        outputs = head(x)
    else:
        raise NotImplementedError()
    if return_model:
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
    else:
        return (inputs, outputs)


class LiteHrNet(tf.keras.models.Model):
    def __init__(self, config: LiteHrnetConfig):
        super().__init__()
        self.stem_m = StemModule(
            stem_channels=config.stem.stem_channels,
            out_channels=config.stem.out_channels,
            name=f"{config.name}.stem"
        )
        branches_chan_list_before = [config.stem.out_channels]
        branches_chan_list_after = []
        self.stages = []
        for spec_idx, spec in enumerate(config.stages):
            stage_modules = []
            branches_chan_list_after = spec.num_channels_list
            stage_modules.append(
                TransitionModule(
                    branches_chan_list_dst=branches_chan_list_after,
                    branches_chan_list_src=branches_chan_list_before,
                    name=f"{config.name}.s{spec_idx}.t"
                )
            )
            for m_idx in range(spec.num_modules):
                stage_modules.append(
                    LiteNaiveHrModule(
                        num_blocks=spec.num_blocks,
                        branches_chan_list=branches_chan_list_after,
                        name=f"{config.name}.s{spec_idx}.m{m_idx}"
                    )
                )

            branches_chan_list_before = branches_chan_list_after
            self.stages.append(stage_modules)

        assert config.head.version in config.head._HEAD_VERSIONS
        if config.head.version == 'v0':
            self.head = lambda x : x
        elif config.head.version == 'v1':
             self.head = lambda x : x[config.head.v1_scale_idx]
        if config.head.version != 'v2':
            raise NotImplementedError()

        head = HrNetHeadV2(num_scales=len(config.stages), output_channels=config.head.v2_out_channels, name=f"{config.name}.head")
        if config.head.version != 'v2':
            raise NotImplementedError()
        self.head = HrNetHeadV2(num_scales=len(config.stages), output_channels=config.head_output_channels, name=f"{config.name}.head")

    def call(self, x):
        x = self.stem_m(x)
        x = [x]
        for stage in self.stages:
            for m in stage:
                x = m(x)
        x = self.head(x)
        return x