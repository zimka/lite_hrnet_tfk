from lite_hrnet_tfk.modules import StemModule, LiteNaiveHrModule, TransitionModule, HrNetHeadV2
from lite_hrnet_tfk.config import LiteHrnetConfig


def build_func_api_net(x, config:LiteHrnetConfig):
    stem_m = StemModule(
        stem_channels=config.stem_spec.stem_channels,
        out_channels=config.stem_spec.out_channels,
        name=f"{config.name}.stem"
    )
    x = stem_m(x)
    x = [x]
    branches_chan_list_before = [config.stem_spec.out_channels]
    branches_chan_list_after = []
    for spec_idx, spec in enumerate(config.stage_specs):
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

    assert config.head_version in LiteHrnetConfig._HEAD_VERSIONS
    if config.head_version != 'v2':
        raise NotImplementedError()

    head = HrNetHeadV2(num_scales=len(config.stage_specs), output_channels=config.head_output_channels, name=f"{config.name}.head")
    x = head(x)
    return x
