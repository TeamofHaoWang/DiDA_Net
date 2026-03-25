# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    Selects frequency indices to keep for representation learning.
    'random': Sample 'modes' number of indices randomly from the spectrum.
    'else': Sample the lowest 'modes' (low-pass filtering).
    """
    max_modes = seq_len // 2 + 1
    modes = min(modes, max_modes)

    if mode_select_method == 'random':
        index = np.random.choice(max_modes, modes, replace=False)
    else:
        index = np.arange(modes)

    index.sort()
    return index.tolist()


class FourierBlock(nn.Module):
    """
    1D Fourier Enhanced Block (FEB).
    Performs FFT -> Frequency Domain Linear Transform -> Inverse FFT.
    Acts as a learnable spectral filter for denoising and feature extraction.
    """

    def __init__(self, in_channels, out_channels, seq_len, modes=16, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Determine frequency indices to process
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)

        # Learnable complex weights for spectral transformation
        # Scale initialized to prevent gradient explosion/vanishing in complex space
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, len(self.index), dtype=torch.cfloat)
        )

        print(f"Fourier Enhanced Block initialized: modes={len(self.index)}, indices={self.index}")

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [Batch, Seq_Len, Channels]
        Returns:
            Output tensor of shape [Batch, Seq_Len, Out_Channels]
        """
        B, L, C = x.shape

        # 1. Permute to [Batch, Channels, Seq_Len] for FFT efficiency
        x = x.permute(0, 2, 1)

        # 2. Compute Real-valued Fast Fourier Transform
        # Output shape: [Batch, Channels, L//2 + 1]
        x_ft = torch.fft.rfft(x, dim=-1)

        # 3. Create output buffer in frequency domain
        out_ft = torch.zeros(B, self.out_channels, L // 2 + 1, device=x.device, dtype=torch.cfloat)

        # 4. Vectorized Spectral Multiplication (Replacing the loop)
        # We extract only the selected modes and multiply by the learnable complex weights
        # x_ft[:, :, self.index] -> [B, C_in, Modes]
        # weights -> [C_in, C_out, Modes]
        # Output -> [B, C_out, Modes]
        selected_modes = x_ft[:, :, self.index]
        transformed_modes = torch.einsum("bim,iom->bom", selected_modes, self.weights)

        # Assign back to the output spectrum buffer
        out_ft[:, :, :len(self.index)] = transformed_modes

        # 5. Inverse FFT back to time domain
        # n=L ensures the length matches the input even if L was odd
        x_out = torch.fft.irfft(out_ft, n=L)

        # 6. Permute back to [Batch, Seq_Len, Channels]
        return x_out.permute(0, 2, 1)