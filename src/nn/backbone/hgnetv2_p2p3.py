"""HGNetv2 backbone truncated to P2 and P3 features."""

import torch.nn as nn

from ...core import register
from .hgnetv2 import HGNetv2

__all__ = ["HGNetv2P2P3"]


@register()
class HGNetv2P2P3(HGNetv2):
    """Return only P2 (stride 4) and P3 (stride 8).

    In HGNetv2 these correspond to Stage1 and Stage2. Stages 3 and 4 are
    removed after initialization, so they consume neither parameters nor
    forward computation.
    """

    def __init__(self, *args, **kwargs):
        # Source indices: 0=Stage1/P2, 1=Stage2/P3,
        #                 2=Stage3/P4, 3=Stage4/P5.
        super().__init__(*args, return_idx=[0, 1], **kwargs)

        selected = [0, 1]
        original_channels = self._out_channels
        original_strides = self._out_strides

        # Stage2 depends on Stage1, so retain exactly the first two stages.
        self.stages = nn.ModuleList(list(self.stages[:2]))
        self._out_channels = [original_channels[i] for i in selected]
        self._out_strides = [original_strides[i] for i in selected]
