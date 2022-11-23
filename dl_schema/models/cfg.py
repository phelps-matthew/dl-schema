"""Model configuration dataclasses."""

from typing import List, Optional, Union
from dataclasses import dataclass, field


@dataclass()
class ResnetConfig:
    """config for ResNet specification"""

    # number of input channels
    in_channels: int = 1
    # kernel size of first layer (7 Hu)
    initial_kernel_size: int = 7
    # stride of first layer (2 Hu)
    initial_stride: int = 2
    # number of output units (1000 Hu)
    num_classes: int = 1000


@dataclass()
class BabyCNNConfig:
    """config for BabyCNN specification"""

    # input image channels
    in_channels: int = 1
    # fully connected units in head hidden layer
    fc_units: int = 128
    # number of output logits in model head
    out_features: int = 10
    # dropout percentage of fc1 layer
    dropout1: float = 0.25
    # dropout percentage of fc2 layer
    dropout2: float = 0.50


@dataclass()
class VGG11Config:
    """config for VGG11 specification"""

    # input image channels
    in_channels: int = 1
    # fully connected units in head hidden layer
    fc_units: int = 128
    # number of output logits in model head
    out_features: int = 10
