import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from typing import List, Tuple, Optional
from functools import partial
from einops import rearrange


class ChannelAdaptiveLayerNorm(nn.Module):
    """Enhanced LayerNorm supporting both channel-order formats with dynamic shape handling."""

    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-6,
                 data_format: str = "channels_first") -> None:
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(normalized_shape)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(normalized_shape)))
        self.eps = eps
        self.data_format = data_format.lower()

        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"Unsupported data format: {data_format}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)

            # channels_first implementation
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class DeployableConvBN(nn.Sequential):
    """Optimizable Conv-BN block with deployment conversion capability."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: Optional[int] = None,
                 dilation: int = 1,
                 groups: int = 1,
                 bn_weight_init: float = 1.0) -> None:
        padding = autopad(kernel_size) if padding is None else padding
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        nn.init.constant_(self[1].weight, bn_weight_init)
        nn.init.constant_(self[1].bias, 0)

    @torch.no_grad()
    def convert_to_deploy(self) -> nn.Conv2d:
        """Fuses BN into Conv for deployment"""
        conv, bn = self[0], self[1]
        fused_conv = nn.Conv2d(
            conv.in_channels * conv.groups,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True
        )

        # Fuse weights
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        fused_conv.weight.data = conv.weight * scale.view(-1, 1, 1, 1)
        fused_conv.bias.data = bn.bias - bn.running_mean * scale
        return fused_conv


class MultiHeadAttentionBlock(nn.Module):
    """Base class for attention mechanisms with shared functionality."""

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 normalize_before: bool = False) -> None:
        super().__init__()
        self.norm1 = ChannelAdaptiveLayerNorm(dim)
        self.norm2 = ChannelAdaptiveLayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.normalize_before = normalize_before


class WindowAttentionEncoder(MultiHeadAttentionBlock):
    """Transformer layer with local window attention."""

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int = 2048,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 window_size: int = 7,
                 normalize_before: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, normalize_before)

        self.attention = LocalWindowAttention(
            dim=embed_dim,
            num_heads=num_heads,
            window_resolution=window_size
        )

        # FFN layers
        self.ffn = nn.Sequential(
            nn.Conv2d(embed_dim, ffn_dim, 1),
            self.activation,
            nn.Dropout(dropout),
            nn.Conv2d(ffn_dim, embed_dim, 1)
        )

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Enhanced forward with optional pre-normalization"""
        if self.normalize_before:
            return self._forward_pre_norm(x, mask, pos_embed)
        return self._forward_post_norm(x, mask, pos_embed)

    def _forward_post_norm(self, x, mask=None, pos_embed=None):
        """Post-normalization forward pass"""
        attn_out = self.attention(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        return self.norm2(x)

    def _forward_pre_norm(self, x, mask=None, pos_embed=None):
        """Pre-normalization forward pass"""
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm)
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        return x + self.dropout(ffn_out)


class PLAIFI(MultiHeadAttentionBlock):
    """Transformer layer with polarized linear attention."""

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int = 2048,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 input_resolution: Tuple[int, int] = (20, 20),
                 normalize_before: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, normalize_before)

        self.attention = PolarizedLinearAttention(
            dim=embed_dim,
            hw=input_resolution,
            num_heads=num_heads
        )

        # FFN layers
        self.ffn = nn.Sequential(
            nn.Conv2d(embed_dim, ffn_dim, 1),
            self.activation,
            nn.Dropout(dropout),
            nn.Conv2d(ffn_dim, embed_dim, 1)
        )

    def forward(self, x, mask=None, pos_embed=None):
        """Unified interface matching WindowAttentionEncoder"""
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)

        if self.normalize_before:
            x_norm = self.norm1(x_flat)
            attn_out = self.attention(x_norm)
        else:
            attn_out = self.attention(x_flat)

        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.dropout(attn_out)

        if not self.normalize_before:
            x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        return self.norm2(x) if not self.normalize_before else x