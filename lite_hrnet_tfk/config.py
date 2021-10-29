from dataclasses import dataclass
from typing import List


@dataclass
class StageSpec:
    num_modules: int
    num_blocks: int
    num_channels_list: List[int]
    with_fuse: bool = True


@dataclass
class StemSpec:
    stem_channels: int = 32
    out_channels: int = 32
    expand_ration: int = 1


@dataclass
class LiteHrnetConfig:
    stage_specs: List[StageSpec]
    stem_spec: StemSpec

    head_output_channels: int
    head_version: str = 'v2'
    _HEAD_VERSIONS = ('v1', 'v2', 'v2p')

    name: str = "LiteHrNet"

    @classmethod
    def naive18(cls, head_output_channels=32):
        return cls(
            stage_specs=[
                StageSpec(num_modules=2, num_blocks=2, num_channels_list=(40, 80)),
                StageSpec(num_modules=4, num_blocks=2, num_channels_list=(40, 80, 160)),
                StageSpec(num_modules=4, num_blocks=2, num_channels_list=(40, 80, 160, 320)),
            ],
            stem_spec=StemSpec(),
            name="LiteHrNetNaive18",
            head_output_channels=head_output_channels
        )
