from dataclasses import dataclass
from typing import List


@dataclass
class StageSpec:
    num_modules: int
    num_blocks: int
    num_channels_list: List[int]


@dataclass
class StemSpec:
    stem_channels: int = 32
    out_channels: int = 32
    expand_ration: int = 1


@dataclass
class HeadSpec:
    _HEAD_VERSIONS = (
        'v0', # None, just forward all scales
        'v1', # Select v1_scale_idx scale and apply conv 1x1
        'v2', # Upsample, concat and apply conv 1x1
        'vi', # Iterative head
    )
    out_channels: int = 32
    version: str = 'v2'
    v1_scale_idx: int = 0


@dataclass
class LiteHrnetConfig:
    stem: StemSpec
    stages: List[StageSpec]
    head: HeadSpec

    name: str = "LiteHrNet"

    @classmethod
    def naive18(cls, out_channels=32):
        """
        https://github.com/HRNet/Lite-HRNet/blob/hrnet/configs/top_down/naive_litehrnet/coco/naive_litehrnet_18_coco_256x192.py#L39
        """
        head = HeadSpec()
        head.out_channels = out_channels
        return cls(
            stem=StemSpec(),
            stages=[
                StageSpec(num_modules=2, num_blocks=2, num_channels_list=(40, 80)),
                StageSpec(num_modules=4, num_blocks=2, num_channels_list=(40, 80, 160)),
                StageSpec(num_modules=2, num_blocks=2, num_channels_list=(40, 80, 160, 320)),
            ],
            name="Naive18",
            head=head
        )
