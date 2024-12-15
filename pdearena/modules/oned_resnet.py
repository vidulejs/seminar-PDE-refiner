from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .activations import ACTIVATION_REGISTRY
from .fourier import SpectralConv1d

import torch
from torch import nn
import torch.nn.functional as F

# Assuming SpectralConv1d is defined elsewhere
# from your_module import SpectralConv1d

class FourierBasicBlock1D(nn.Module):
    """Basic block for 1D Fourier Neural Operator ResNet.

    Args:
        in_planes (int): Number of input channels.
        planes (int): Number of output channels.
        modes (int): Number of Fourier modes to keep.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        modes: int = 16,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        self.modes = modes
        self.fourier1 = SpectralConv1d(in_planes, planes, modes=self.modes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, padding=0, bias=True)

        self.activation = getattr(F, activation)

    def forward(self, x):
        x1 = self.fourier1(x)
        x2 = self.conv1(x)
        out = self.activation(x1 + x2)

        return out

class ResNet1D(nn.Module):
    """1D ResNet with Fourier layers (FNO).

    Args:
        n_input_scalar_components (int): Number of input scalar components.
        n_output_scalar_components (int): Number of output scalar components.
        block (nn.Module): Block type to use (e.g., FourierBasicBlock1D).
        num_blocks (list of int): Number of blocks in each layer.
        time_history (int): Number of time steps in the input.
        time_future (int): Number of time steps to predict.
        hidden_channels (int): Number of channels in hidden layers.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        modes (int): Number of Fourier modes to keep.
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        block,
        num_blocks: list,
        time_history: int,
        time_future: int,
        hidden_channels: int = 64,
        activation: str = "gelu",
        norm: bool = False,
        modes: int = 16,
    ):
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_output_scalar_components = n_output_scalar_components
        self.in_planes = hidden_channels
        self.time_history = time_history
        self.time_future = time_future

        insize = time_history * n_input_scalar_components
        self.conv_in1 = nn.Conv1d(
            insize,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_in2 = nn.Conv1d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out1 = nn.Conv1d(
            self.in_planes,
            self.in_planes,
            kernel_size=1,
            bias=True,
        )
        self.conv_out2 = nn.Conv1d(
            self.in_planes,
            time_future * n_output_scalar_components,
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    block,
                    self.in_planes,
                    num_blocks[i],
                    activation=activation,
                    norm=norm,
                    modes=modes,
                )
                for i in range(len(num_blocks))
            ]
        )
        self.activation = getattr(F, activation)

    def _make_layer(self, block, planes, num_blocks, activation, norm, modes):
        layers = []
        for _ in range(num_blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    modes=modes,
                    activation=activation,
                    norm=norm,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, time_history, n_input_scalar_components, width)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(-1))  # Merge time and input_channels
        x = self.activation(self.conv_in1(x))
        x = self.activation(self.conv_in2(x))

        for layer in self.layers:
            x = layer(x)

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        x = x.view(
            batch_size,
            self.time_future,
            self.n_output_scalar_components,
            x.size(-1),
        )
        return x