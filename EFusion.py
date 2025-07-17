import torch
from torch import nn, Tensor
from typing import List
from .modules.conv import Conv


class EFusion(nn.Module):
    """Fuses multiple input channels through convolutional operations.

    Args:
        input_channels: List of input channel dimensions (e.g., [64, 128])
        output_channels: Desired output channel dimension
    """

    def __init__(self, input_channels: List[int], output_channels: int) -> None:
        super().__init__()

        # Configuration
        self.hidden_dim = output_channels // 2

        # Layer definitions
        self.channel_fusion = self._build_channel_fusion(sum(input_channels))
        self.feature_extractor = self._build_feature_extractor()
        self.output_projection = self._build_output_projection()

    def _build_channel_fusion(self, total_input_channels: int) -> nn.Module:
        """1x1 conv for channel fusion"""
        return Conv(total_input_channels, self.hidden_dim, kernel_size=1)

    def _build_feature_extractor(self) -> nn.Module:
        """3x3 conv for spatial feature extraction"""
        return Conv(self.hidden_dim, self.hidden_dim, kernel_size=3)

    def _build_output_projection(self) -> nn.Module:
        """Final 1x1 conv for dimension adjustment"""
        return Conv(self.hidden_dim, self.hidden_dim * 2, kernel_size=1)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        """Processing pipeline:
        1. Concatenate all input tensors
        2. Channel fusion
        3. Feature extraction
        4. Output projection
        """
        concatenated = torch.cat(inputs, dim=1)
        fused = self.channel_fusion(concatenated)
        features = self.feature_extractor(fused)
        return self.output_projection(features)