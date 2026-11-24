"""The ARISE-DETR encoder and its residual detail-semantic fusion."""

import torch
import torch.nn as nn

from ...core import register
from .hybrid_encoder import ConvNormLayer_fuse, VGGBlock
from .hybrid_encoder_bio import channel_shuffle
from .hybrid_encoder_bio2 import BioFusionV2
from .hybrid_encoder_biody import HybridEncoderBioDy


class RGCSPELANLite(nn.Module):
    """D-FINE-native RepGhost CSP-ELAN with n=1, scale=e=0.5."""

    def __init__(self, c1, c2, act="silu"):
        super().__init__()
        hidden = c2 // 2
        mid = hidden // 2
        self.hidden = hidden
        self.cv1 = ConvNormLayer_fuse(c1, 2 * hidden, 1, 1, act=act)
        self.rep = VGGBlock(hidden, mid, act=nn.SiLU())
        self.tail = ConvNormLayer_fuse(mid, mid, 1, 1, act=act)
        self.cv2 = ConvNormLayer_fuse(hidden + 2 * mid, c2, 1, 1, act=act)

    def forward(self, x):
        identity, transformed = self.cv1(x).chunk(2, dim=1)
        transformed = self.rep(transformed)
        tail = self.tail(transformed)
        return self.cv2(torch.cat([identity, transformed, tail], dim=1))


class FusionRNoGate(BioFusionV2):
    """Fuse the P2 detail stream and the upsampled P3 semantic stream."""
    def __init__(self, channels=256, ratio=0.5, act='silu'):
        super().__init__(channels=channels, ratio=ratio, groups=1, act=act)
        self.rgc = RGCSPELANLite(channels, channels, act=act)
        # These branches/parameters are not part of the final fusion forward path.
        del self.channel_gate, self.spatial_gate
        del self.beta_c, self.beta_s
        del self.local, self.context, self.fuse

    def forward(self, x):
        semantic, detail = torch.split(x, [self.channels, self.channels], dim=1)
        detail = self.detail(detail)          # P2: 1x1 projection + PConv
        semantic = self.semantic(semantic)    # P3: 1x1 projection + InceptionDW
        merged = torch.cat([detail, semantic], dim=1)
        base = self.base(channel_shuffle(merged, 2))
        return base + self.alpha * self.rgc(merged)


@register()
class ARISEEncoder(HybridEncoderBioDy):
    """Top-down propagation with content-aware upsampling and residual fusion."""
    def __init__(self, *args, bio_ratio=0.5, **kwargs):
        super().__init__(*args, bio_ratio=bio_ratio, **kwargs)
        self.fpn_blocks[-1] = FusionRNoGate(
            channels=self.hidden_dim,
            ratio=bio_ratio,
            act=kwargs.get('act', 'silu'),
        )
