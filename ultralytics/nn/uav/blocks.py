import torch
from torch import nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv

# 用于构建ResNet18
class BasicBlock(nn.Module):
    """ResNet BasicBlock with standard convolution layers."""

    def __init__(self, in_channels, out_channels, stride=1):
        """Initialize convolution with given parameters."""
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels, k=3, s=stride, p=1, act=True)
        self.cv2 = Conv(out_channels, out_channels, k=3, s=1, p=1, act=False)
        self.shortcut = nn.Sequential(Conv(in_channels, out_channels, k=1, s=stride, act=False)) if stride != 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv2(self.cv1(x)) + self.shortcut(x))
    
class ResNetLayer_Basic(nn.Module):
    """ResNet layer with multiple ResNet BasicBlock."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [BasicBlock(c1, c2, s)]
            blocks.extend([BasicBlock(c2, c2, 1) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)
    


