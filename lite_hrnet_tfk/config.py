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
class HeadSpec:
    _HEAD_VERSIONS = (
        'v0', # None, just forward all scales
        'v1', # Forward scale with the highest resolution
        'v2', # Upsample, concat and apply conv 1x1
        'v2p'
    )
    v2_out_channels: int = 32
    version: str = 'v2'
    v1_scale_idx: int = 0



@dataclass
class LiteHrnetConfig:
    stages: List[StageSpec]
    stem: StemSpec
    head: HeadSpec

    name: str = "LiteHrNet"

    @classmethod
    def naive18(cls, head_output_channels=32):
        return cls(
            stages=[
                StageSpec(num_modules=2, num_blocks=2, num_channels_list=(40, 80)),
                StageSpec(num_modules=4, num_blocks=2, num_channels_list=(40, 80, 160)),
                StageSpec(num_modules=4, num_blocks=2, num_channels_list=(40, 80, 160, 320)),
            ],
            stem=StemSpec(),
            name="LiteHrNetNaive18",
            head=HeadSpec()
        )
