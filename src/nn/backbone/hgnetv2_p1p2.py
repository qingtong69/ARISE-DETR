"""HGNetv2 backbone truncated to its two highest-resolution features."""

import torch
import torch.nn.functional as F

from ...core import register
from .hgnetv2 import HGNetv2

__all__ = ["HGNetv2P1P2"]


@register()
class HGNetv2P1P2(HGNetv2):
    """Return only P1 (stride 2) and P2 (stride 4) from HGNetv2.

    P1 is the output of the first stem convolution. P2 is the output of
    stage1.  The parent constructor first loads the complete official
    HGNetv2 checkpoint; stages 2--4 are then removed, so their parameters
    and computation are not retained by this backbone.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, return_idx=[0], **kwargs)
        self.stages = torch.nn.ModuleList([self.stages[0]])

        self._out_strides = [2, 4]
        self._out_channels = [self.stem.stem1.bn.num_features, self._out_channels[0]]

    def forward(self, x):
        # Keep this sequence identical to StemBlock.forward, while exposing
        # stem1 as P1 before the remaining stem operations construct P2.
        p1 = self.stem.stem1(x)

        x = F.pad(p1, (0, 1, 0, 1))
        x2 = self.stem.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem.stem2b(x2)
        x1 = self.stem.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem.stem3(x)
        x = self.stem.stem4(x)

        p2 = self.stages[0](x)
        return [p1, p2]
