import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from .modules.conv import Conv


class SobelEdgeDetector(nn.Module):
    """3D Sobel operator for edge detection with fixed weights.

    Args:
        channels: Number of input/output channels
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        # Sobel kernels (Y-axis and X-axis variants)
        self._init_sobel_kernels(channels)

    def _init_sobel_kernels(self, channels: int) -> None:
        """Initialize non-trainable Sobel kernels"""
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_x = sobel_y.T

        # Create 3D convolution layers
        self.conv_x = self._create_fixed_conv3d(channels, sobel_x)
        self.conv_y = self._create_fixed_conv3d(channels, sobel_y)

    def _create_fixed_conv3d(self,
                             channels: int,
                             kernel: np.ndarray) -> nn.Conv3d:
        """Helper to create frozen Conv3D layer"""
        kernel_tensor = torch.tensor(kernel, dtype=torch.float32)
        kernel_tensor = kernel_tensor.unsqueeze(0).expand(channels, 1, 1, 3, 3)

        conv = nn.Conv3d(
            channels, channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )
        conv.weight.data = kernel_tensor.clone()
        conv.requires_grad_(False)
        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Sobel operators and combine results"""
        x_3d = x.unsqueeze(2)  # Add dummy dimension for 3D conv
        edge_x = self.conv_x(x_3d)
        edge_y = self.conv_y(x_3d)
        return (edge_x + edge_y).squeeze(2)


class PREFP(nn.Module):
    """Pyramid-style edge feature generator with:
    - Edge detection
    - Multi-scale pooling
    - Channel projection

    Args:
        input_channels: Input feature channels
        output_channels: List of output channels for each scale
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: List[int]) -> None:
        super().__init__()

        # Edge detection module (interchangeable implementations)
        self.edge_detector = SobelEdgeDetector(input_channels)
        # self.edge_detector  = PrewittEdgeDetector(input_channels)
        # self.edge_detector  = RobertEdgeDetector(input_channels)

        # Multi-scale processing
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.projections = self._build_projections(input_channels, output_channels)

    def _build_projections(self,
                           in_ch: int,
                           out_chs: List[int]) -> nn.ModuleList:
        """Create 1x1 convs for channel projection"""
        return nn.ModuleList(
            Conv(in_ch, out_ch, 1) for out_ch in out_chs
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Generate multi-scale edge features:
        1. Base edge map
        2. Downsampled versions
        3. Channel projection
        """
        # Generate edge maps at different scales
        edge_maps = [self.edge_detector(x)]
        for _ in range(len(self.projections)):
            edge_maps.append(self.downsample(edge_maps[-1]))

            # Apply channel projections (skip the original scale)
        return [
            conv(features)
            for conv, features in zip(self.projections, edge_maps[1:])
        ]