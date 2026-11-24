"""Four-level top-down FPN encoder without a bottom-up PAN path."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .hybrid_encoder import CSPLayer, ConvNormLayer_fuse, HybridEncoder, VGGBlock

__all__ = ["HybridEncoderP2P5FPNOnly"]


class DepthwiseSeparableConv(nn.Module):
    """Spatial filtering with a cheap depthwise 3x3 and dense 1x1 mixing."""

    def __init__(self, channels, bias=False, act="silu"):
        super().__init__()
        self.depthwise = ConvNormLayer_fuse(
            channels,
            channels,
            3,
            1,
            g=channels,
            bias=bias,
            act=act,
        )
        self.pointwise = ConvNormLayer_fuse(
            channels, channels, 1, 1, bias=bias, act=act
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class LiteRepNCSPELAN4(nn.Module):
    """Accuracy-oriented lightweight RepNCSPELAN4 for high-resolution P2.

    The CSP/ELAN topology and dense entry/exit projections are retained. Only
    the two branch-tail 3x3 convolutions are made depthwise separable. The
    intermediate width is configurable so the expensive P2 block can use C
    channels instead of the original 2C without narrowing its C-channel output.
    """

    def __init__(self, c1, c2, c3, c4, n=1, bias=False, act="silu"):
        super().__init__()
        if c3 % 2:
            raise ValueError("LiteRepNCSPELAN4 requires an even c3")
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(
                c3 // 2,
                c4,
                n,
                1,
                bias=bias,
                act=act,
                bottletype=VGGBlock,
            ),
            DepthwiseSeparableConv(c4, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(
                c4,
                c4,
                n,
                1,
                bias=bias,
                act=act,
                bottletype=VGGBlock,
            ),
            DepthwiseSeparableConv(c4, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(
            c3 + 2 * c4, c2, 1, 1, bias=bias, act=act
        )

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        y.extend(module(y[-1]) for module in (self.cv2, self.cv3))
        return self.cv4(torch.cat(y, dim=1))


@register()
class HybridEncoderP2P5FPNOnly(HybridEncoder):
    """Fuse P5->P4->P3->P2 and return all four levels to D-FINE."""

    def __init__(
        self,
        *args,
        lite_p2_fusion=False,
        p2_intermediate_ratio=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if len(self.in_channels) != 4:
            raise ValueError("HybridEncoderP2P5FPNOnly requires P2--P5 inputs")

        # Remove every bottom-up path and its parameters.
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        self.out_channels = [self.hidden_dim] * 4
        self.out_strides = list(self.feat_strides)

        # P3->P2 operates at 160x160 for a 640x640 image and dominates the
        # encoder FLOPs. Keep lower-resolution fusion blocks unchanged and
        # selectively replace only this final, highest-resolution block.
        if lite_p2_fusion:
            c = self.hidden_dim
            c3 = max(2, round(c * p2_intermediate_ratio))
            c3 += c3 % 2
            c4 = max(1, round(kwargs.get("expansion", 1.0) * c / 2))
            n = max(1, round(3 * kwargs.get("depth_mult", 1.0)))
            self.fpn_blocks[-1] = LiteRepNCSPELAN4(
                2 * c,
                c,
                c3,
                c4,
                n=n,
                act=kwargs.get("act", "silu"),
            )

    def forward(self, feats):
        if len(feats) != 4:
            raise ValueError("Expected exactly [P2, P3, P4, P5]")

        proj_feats = [
            self.input_proj[index](feature)
            for index, feature in enumerate(feats)
        ]

        if self.num_encoder_layers > 0:
            for encoder_index, feature_index in enumerate(self.use_encoder_idx):
                height, width = proj_feats[feature_index].shape[2:]
                flattened = proj_feats[feature_index].flatten(2).permute(0, 2, 1)

                if self.training or self.eval_spatial_size is None:
                    position = self.build_2d_sincos_position_embedding(
                        width, height, self.hidden_dim, self.pe_temperature
                    ).to(flattened.device)
                else:
                    position = getattr(
                        self, f"pos_embed{feature_index}", None
                    ).to(flattened.device)

                memory = self.encoder[encoder_index](
                    flattened, pos_embed=position
                )
                proj_feats[feature_index] = (
                    memory.permute(0, 2, 1)
                    .reshape(-1, self.hidden_dim, height, width)
                    .contiguous()
                )

        # Pure top-down FPN: P5 -> P4 -> P3 -> P2.
        inner_outs = [proj_feats[-1]]
        for feature_index in range(3, 0, -1):
            block_index = 3 - feature_index
            high = self.lateral_convs[block_index](inner_outs[0])
            inner_outs[0] = high
            fused = self.fpn_blocks[block_index](
                torch.cat(
                    [
                        F.interpolate(high, scale_factor=2.0, mode="nearest"),
                        proj_feats[feature_index - 1],
                    ],
                    dim=1,
                )
            )
            inner_outs.insert(0, fused)

        # Ascending stride order: fused P2, fused P3, fused P4, lateral P5.
        return inner_outs
