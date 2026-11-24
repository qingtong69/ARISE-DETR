"""HGNetv2 wrapper returning P2, P3, P4 and P5."""

from ...core import register
from .hgnetv2 import HGNetv2

__all__ = ["HGNetv2P2P5"]


@register()
class HGNetv2P2P5(HGNetv2):
    """Return B0 features with channels [64, 256, 512, 1024]."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, return_idx=[0, 1, 2, 3], **kwargs)
