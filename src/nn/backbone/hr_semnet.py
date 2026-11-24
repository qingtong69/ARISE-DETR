import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .common import ConvNormLayer

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = ConvNormLayer(
            c1,
            c2,
            kernel_size = k,
            stride=s,
            padding=p,
            act="silu",
        )

    def forward(self, x):
        return self.conv(x)

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()

        self.cv1 = ConvBNAct(c1, c2, k=3, s=1)
        self.cv2 = ConvBNAct(c2, c2, k=3, s=1)
        self.use_shortcut = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))

        if self.use_shortcut:
            y = y + x

        return y

class CARAFE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.inc = inC
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(inC, inC // 4, 1)
        if self.kernel_size % 2 == 0:
            self.encoder = nn.Conv2d(inC // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                    self.kernel_size, 1, 0)
        else:
            self.encoder = nn.Conv2d(inC // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                    self.kernel_size, 1, self.kernel_size // 2)
        # self.out = DSConv(inC, outC, 3, 1)

    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(in_tensor)  # (N, Cm, H, W)
        if self.kernel_size % 2 == 0:
            kernel_tensor = F.pad(kernel_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2 - 1,
                                              self.kernel_size // 2, self.kernel_size // 2 - 1),
                            mode='constant', value=0)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2) # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        if self.kernel_size % 2 == 0:
            in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2 - 1,
                                            self.kernel_size // 2, self.kernel_size // 2 - 1),
                            mode='constant', value=0)
        else:
            in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                            self.kernel_size // 2, self.kernel_size // 2),
                            mode='constant', value=0) # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1) # (N, C, H, W+Kup//2+Kup//2, Kup)
        in_tensor = in_tensor.unfold(3, self.kernel_size, step=1) # (N, C, H, W, Kup, Kup)
        in_tensor = in_tensor.reshape(N, C, H, W, -1) # (N, C, H, W, Kup^2)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(in_tensor, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)

        out_tensor = out_tensor.permute(0, 3, 1, 2)

        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)

        # out_tensor = self.out(out_tensor)
        return out_tensor


class bilinear(nn.Module):
    def __init__(self, c1, c2, u=2):
        super().__init__()
        self.bool = False
        self.up = u
        if c1 != c2:
            self.bool = True
            self.cv = Conv(c1, c2, 1, 1)
        self.upsample = nn.Upsample(None, self.up, 'bilinear')

    def forward(self, x):
        if self.bool:
            x = self.cv(x)
        # _, _, H, W = x.size()
        # x = F.interpolate(
        #     x,
        #     size=(H*self.up, W*self.up),
        #     mode='bilinear',
        #     align_corners=True)
        x = self.upsample(x)

        return x

class LCSM(nn.Module):
    def __init__(self, channels, kernel_size=1, factor=2):
        super().__init__()

        self.factor = factor
        self.channel_conv = ConvBNAct(
            channels * factor**2,
            channels,
            k=1,
            s=1,
        )
        self.carafe = CARAFE(
            channels,
            channels,
            kernel_size=kernel_size,
            up_factor=factor,
        )

    def forward(self, x):
        h, w = x.shape[-2:]

        if h % self.factor != 0 or w % self.factor != 0:
            raise ValueError(
                f"Feature size {(h, w)} must be divisible "
                f"by factor={self.factor}"
            )
        identity = x

        x = F.pixel_unshuffle(x, self.factor)
        x = self.channel_conv(x)
        x = self.carafe(x)

        return x + identity

class CLfS3(nn.Module):
    """
    LCSM-enhanced C2f with three Bottleneck stages.

    Used for P1:
        Bottleneck
        -> LCSM(factor=2) -> Bottleneck
        -> LCSM(factor=4) -> Bottleneck
        -> concatenate all branches
        -> 1x1 convolution
    """

    def __init__(
        self,
        c1,
        c2,
        shortcut=False,
        expansion=0.5,
    ):
        super().__init__()

        hidden_channels = int(c2 * expansion)

        if hidden_channels <= 0:
            raise ValueError(
                f"Invalid hidden channels: c2={c2}, expansion={expansion}"
            )

        self.hidden_channels = hidden_channels

        # C2f输入通道划分成两个分支
        self.cv1 = ConvBNAct(
            c1,
            2 * hidden_channels,
            k=1,
            s=1,
        )

        self.bottleneck1 = Bottleneck(
            hidden_channels,
            hidden_channels,
            shortcut=shortcut,
        )
        self.bottleneck2 = Bottleneck(
            hidden_channels,
            hidden_channels,
            shortcut=shortcut,
        )
        self.bottleneck3 = Bottleneck(
            hidden_channels,
            hidden_channels,
            shortcut=shortcut,
        )

        # P1使用两级局部语义增强
        self.lcsm2 = LCSM(
            hidden_channels,
            kernel_size=1,
            factor=2,
        )
        self.lcsm4 = LCSM(
            hidden_channels,
            kernel_size=2,
            factor=4,
        )

        # 拼接y、y0、y1、y2、y3，共5个分支
        self.cv2 = ConvBNAct(
            5 * hidden_channels,
            c2,
            k=1,
            s=1,
        )

    def forward(self, x):
        # [B, 2*hidden, H, W]
        x = self.cv1(x)

        # 两个通道分支，每个都是hidden_channels
        y, y0 = x.chunk(2, dim=1)

        # 第一层只执行Bottleneck
        y1 = self.bottleneck1(y0)

        # 第二层先进行factor=2的局部语义增强
        y2 = self.lcsm2(y1)
        y2 = self.bottleneck2(y2)

        # 第三层使用factor=4的局部语义增强
        y3 = self.lcsm4(y2)
        y3 = self.bottleneck3(y3)

        # C2f式多阶段特征拼接
        out = torch.cat(
            (y, y0, y1, y2, y3),
            dim=1,
        )

        return self.cv2(out)

class CLfS4(nn.Module):
    """
    LCSM-enhanced C2f with four Bottleneck stages.

    Used for P2:
        Bottleneck
        -> LCSM(factor=2) -> Bottleneck
        -> LCSM(factor=4) -> Bottleneck
        -> LCSM(factor=8) -> Bottleneck
        -> concatenate all branches
        -> 1x1 convolution
    """

    def __init__(
        self,
        c1,
        c2,
        shortcut=False,
        expansion=0.5,
    ):
        super().__init__()

        hidden_channels = int(c2 * expansion)

        if hidden_channels <= 0:
            raise ValueError(
                f"Invalid hidden channels: c2={c2}, expansion={expansion}"
            )

        self.hidden_channels = hidden_channels

        # 输入投影并拆分为两个分支
        self.cv1 = ConvBNAct(
            c1,
            2 * hidden_channels,
            k=1,
            s=1,
        )

        self.bottleneck1 = Bottleneck(
            hidden_channels,
            hidden_channels,
            shortcut=shortcut,
        )
        self.bottleneck2 = Bottleneck(
            hidden_channels,
            hidden_channels,
            shortcut=shortcut,
        )
        self.bottleneck3 = Bottleneck(
            hidden_channels,
            hidden_channels,
            shortcut=shortcut,
        )
        self.bottleneck4 = Bottleneck(
            hidden_channels,
            hidden_channels,
            shortcut=shortcut,
        )

        # P2使用三种局部窗口尺度
        self.lcsm2 = LCSM(
            hidden_channels,
            kernel_size=1,
            factor=2,
        )
        self.lcsm4 = LCSM(
            hidden_channels,
            kernel_size=2,
            factor=4,
        )
        self.lcsm8 = LCSM(
            hidden_channels,
            kernel_size=3,
            factor=8,
        )

        # 拼接y、y0、y1、y2、y3、y4，共6个分支
        self.cv2 = ConvBNAct(
            6 * hidden_channels,
            c2,
            k=1,
            s=1,
        )

    def forward(self, x):
        # [B, 2*hidden, H, W]
        x = self.cv1(x)

        y, y0 = x.chunk(2, dim=1)

        # 第一层
        y1 = self.bottleneck1(y0)

        # factor=2
        y2 = self.lcsm2(y1)
        y2 = self.bottleneck2(y2)

        # factor=4
        y3 = self.lcsm4(y2)
        y3 = self.bottleneck3(y3)

        # factor=8
        y4 = self.lcsm8(y3)
        y4 = self.bottleneck4(y4)

        out = torch.cat(
            (y, y0, y1, y2, y3, y4),
            dim=1,
        )

        return self.cv2(out)

@register()
class HRSemNetBackbone(nn.Module):
    """
    HR-SemNet high-resolution backbone.

    Input:
        x: [B, 3, H, W]

    Output:
        P1: [B, p1_channels, H/2, W/2]
        P2: [B, p2_channels, H/4, W/4]

    The backbone performs:
        1. Initial P1 and P2 extraction
        2. First P2 -> P1 -> P2 fusion
        3. Second P2 -> P1 -> P2 fusion
    """

    def __init__(
        self,
        p1_channels=32,
        p2_stem_channels=96,
        p2_channels=128,
        return_idx=(0, 1),
    ):
        super().__init__()

        if not return_idx:
            raise ValueError("return_idx cannot be empty")

        if any(i not in (0, 1) for i in return_idx):
            raise ValueError(
                f"return_idx must contain only 0 or 1, got {return_idx}"
            )

        self.return_idx = list(return_idx)
        self.p1_channels = p1_channels
        self.p2_stem_channels = p2_stem_channels
        self.p2_channels = p2_channels

        # 0=P1，1=P2
        self._out_channels = [
            p1_channels,
            p2_channels,
        ]
        self._out_strides = [
            2,
            4,
        ]

        self.out_channels = [
            self._out_channels[i]
            for i in self.return_idx
        ]
        self.out_strides = [
            self._out_strides[i]
            for i in self.return_idx
        ]

        # --------------------------------------------------
        # Initial P1
        #
        # Image
        #   -> Conv(stride=2)
        #   -> CLfS3
        #   -> P1 initial
        # --------------------------------------------------
        self.p1_stem = ConvBNAct(
            c1=3,
            c2=p1_channels,
            k=3,
            s=2,
        )

        self.p1_initial_block = CLfS3(
            c1=p1_channels,
            c2=p1_channels,
            shortcut=False,
            expansion=0.5,
        )

        # --------------------------------------------------
        # Initial P2
        #
        # P1
        #   -> Conv(stride=2)
        #   -> CLfS4
        #   -> P2 initial
        # --------------------------------------------------
        self.p2_stem = ConvBNAct(
            c1=p1_channels,
            c2=p2_stem_channels,
            k=3,
            s=2,
        )

        self.p2_initial_block = CLfS4(
            c1=p2_stem_channels,
            c2=p2_stem_channels,
            shortcut=True,
            expansion=0.5,
        )

        # ==================================================
        # First P2 -> P1 -> P2 fusion
        # ==================================================

        # Refine initial P2
        self.stage1_p2_refine = CLfS4(
            c1=p2_stem_channels,
            c2=p2_stem_channels,
            shortcut=True,
            expansion=0.5,
        )

        # Project P2 channels before upsampling to P1
        self.stage1_p2_to_p1_proj = ConvBNAct(
            c1=p2_stem_channels,
            c2=p1_channels,
            k=1,
            s=1,
        )

        # Concatenate:
        # upsampled P2: p1_channels
        # initial P1:   p1_channels
        # total:        2 * p1_channels
        self.stage1_p1_fuse = CLfS3(
            c1=2 * p1_channels,
            c2=p1_channels,
            shortcut=False,
            expansion=0.5,
        )

        # Downsample the new P1
        #
        # Original HR-SemNet uses:
        # Conv(out_channels=96, kernel=2, stride=2, padding=0)
        self.stage1_p1_down = ConvBNAct(
            c1=p1_channels,
            c2=p2_stem_channels,
            k=2,
            s=2,
            p=0,
        )

        # Concatenate:
        # downsampled P1: p2_stem_channels
        # refined P2:     p2_stem_channels
        self.stage1_p2_fuse = CLfS4(
            c1=2 * p2_stem_channels,
            c2=p2_channels,
            shortcut=True,
            expansion=0.5,
        )

        # ==================================================
        # Second P2 -> P1 -> P2 fusion
        # ==================================================

        self.stage2_p2_refine = CLfS4(
            c1=p2_channels,
            c2=p2_channels,
            shortcut=True,
            expansion=0.5,
        )

        self.stage2_p2_to_p1_proj = ConvBNAct(
            c1=p2_channels,
            c2=p1_channels,
            k=1,
            s=1,
        )

        # Concatenate:
        # upsampled P2: p1_channels
        # stage1 P1:    p1_channels
        self.stage2_p1_fuse = CLfS3(
            c1=2 * p1_channels,
            c2=p1_channels,
            shortcut=False,
            expansion=0.5,
        )

        # Original HR-SemNet second downsampling outputs 128 channels
        self.stage2_p1_down = ConvBNAct(
            c1=p1_channels,
            c2=p2_channels,
            k=2,
            s=2,
            p=0,
        )

        # Concatenate:
        # downsampled P1: p2_channels
        # refined P2:     p2_channels
        self.stage2_p2_fuse = CLfS4(
            c1=2 * p2_channels,
            c2=p2_channels,
            shortcut=True,
            expansion=0.5,
        )

    @staticmethod
    def _upsample_to(source, target):
        """
        Upsample source to the exact spatial size of target.

        Using target.shape avoids one-pixel mismatches when the input
        image has an unusual size.
        """
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _check_input_size(x):
        """
        CLfS4 uses Pixel-Unshuffle with factor=8 on a stride-4
        feature. Therefore, input H and W should be divisible by 32.
        """
        h, w = x.shape[-2:]

        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(
                f"Input size {(h, w)} must be divisible by 32 "
                "for CLfS4 factor=8 Pixel-Unshuffle"
            )

    def forward(self, x):
        self._check_input_size(x)

        # --------------------------------------------------
        # Initial P1 and P2
        # --------------------------------------------------
        p1_initial = self.p1_stem(x)
        p1_initial = self.p1_initial_block(p1_initial)

        p2_initial = self.p2_stem(p1_initial)
        p2_initial = self.p2_initial_block(p2_initial)

        # --------------------------------------------------
        # First fusion: P2 -> P1
        # --------------------------------------------------
        p2_stage1 = self.stage1_p2_refine(p2_initial)

        p2_up = self.stage1_p2_to_p1_proj(p2_stage1)
        p2_up = self._upsample_to(
            source=p2_up,
            target=p1_initial,
        )

        p1_stage1 = torch.cat(
            (p2_up, p1_initial),
            dim=1,
        )
        p1_stage1 = self.stage1_p1_fuse(p1_stage1)

        # --------------------------------------------------
        # First fusion: P1 -> P2
        # --------------------------------------------------
        p1_down = self.stage1_p1_down(p1_stage1)

        if p1_down.shape[-2:] != p2_stage1.shape[-2:]:
            raise RuntimeError(
                "Stage-1 fusion size mismatch: "
                f"P1 downsampled={p1_down.shape[-2:]}, "
                f"P2={p2_stage1.shape[-2:]}"
            )

        p2_stage1 = torch.cat(
            (p1_down, p2_stage1),
            dim=1,
        )
        p2_stage1 = self.stage1_p2_fuse(p2_stage1)

        # --------------------------------------------------
        # Second fusion: P2 -> P1
        # --------------------------------------------------
        p2_stage2 = self.stage2_p2_refine(p2_stage1)

        p2_up = self.stage2_p2_to_p1_proj(p2_stage2)
        p2_up = self._upsample_to(
            source=p2_up,
            target=p1_stage1,
        )

        p1_stage2 = torch.cat(
            (p2_up, p1_stage1),
            dim=1,
        )
        p1_stage2 = self.stage2_p1_fuse(p1_stage2)

        # --------------------------------------------------
        # Second fusion: P1 -> P2
        # --------------------------------------------------
        p1_down = self.stage2_p1_down(p1_stage2)

        if p1_down.shape[-2:] != p2_stage2.shape[-2:]:
            raise RuntimeError(
                "Stage-2 fusion size mismatch: "
                f"P1 downsampled={p1_down.shape[-2:]}, "
                f"P2={p2_stage2.shape[-2:]}"
            )

        p2_stage2 = torch.cat(
            (p1_down, p2_stage2),
            dim=1,
        )
        p2_stage2 = self.stage2_p2_fuse(p2_stage2)

        # Final features
        p1 = p1_stage2
        p2 = p2_stage2

        features = [p1, p2]

        return [
            features[i]
            for i in self.return_idx
        ]