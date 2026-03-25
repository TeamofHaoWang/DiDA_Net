# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv2d, ModuleList


class GWTNet_layer(nn.Module):
    """
    Graph-WaveNet Layer variation.
    Combines Dilated Causal Convolutions with Graph Convolutions.
    Uses Skip Connections to aggregate multi-scale temporal features.
    """

    def __init__(self, num_nodes, in_dim, out_dim, d_model, d_ff,
                 dropout=0.3, kernel_size=2, blocks=3, layers=1, args=None):
        super().__init__()
        self.blocks = blocks
        self.layers = layers
        self.args = args

        # Mapping input features to hidden dimension
        self.start_conv = nn.Conv2d(in_dim, d_model, kernel_size=(1, 1))

        depth = blocks * layers
        self.supports_len = 1  # Standard GraphConv usually takes 1 or 2 supports

        # Module collections for WaveNet structure
        self.residual_convs = ModuleList([Conv2d(d_model, d_model, (1, 1)) for _ in range(depth)])
        self.skip_convs = ModuleList([Conv2d(d_model, d_ff, (1, 1)) for _ in range(depth)])
        self.bn = ModuleList([BatchNorm2d(d_model) for _ in range(depth)])
        self.graph_convs = ModuleList([
            GraphConvNet(d_model, d_model, dropout, support_len=self.supports_len)
            for _ in range(depth)
        ])

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()

        # Calculate receptive field based on dilation and kernel size
        receptive_field = 1
        for b in range(blocks):
            dilation = 1
            for l in range(layers):
                # Dilated convolutions (Filter and Gate)
                self.filter_convs.append(Conv2d(d_model, d_model, (1, kernel_size), dilation=dilation))
                self.gate_convs.append(Conv2d(d_model, d_model, (1, kernel_size), dilation=dilation))
                receptive_field += (kernel_size - 1) * dilation
                dilation *= 2

        self.receptive_field = receptive_field
        self.output_bn = BatchNorm2d(d_model)

    def forward(self, x, adj, **kwargs):
        """
        Input x: [Batch, Channels, Nodes, Seq_Len]
        Input adj: List of adjacency matrices
        """
        x = self.start_conv(x)

        # Pre-pad input to satisfy the receptive field requirement
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))

        skip = 0
        for i in range(self.blocks * self.layers):
            # 1. Residual branch
            residual = x

            # 2. Dilated Gated Convolution
            # (Note: In this specific version, filter/gate are applied before GraphConv)
            # You might need to add padding here if kernel_size > 1 to maintain length
            filt = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filt * gate

            # 3. Skip Connection: Capture features from current scale
            s = self.skip_convs[i](x)
            if isinstance(skip, torch.Tensor):
                # Align temporal dimensions for skip summation
                skip = s + skip[:, :, :, -s.size(3):]
            else:
                skip = s

            # Final layer check
            if i == (self.blocks * self.layers - 1):
                break

            # 4. Graph Convolution: Spatial Message Passing
            x = self.graph_convs[i](x, adj)

            # 5. Residual Connection
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        # Return the aggregated skip features
        x = self.output_bn(skip)
        return x.transpose(1, -1)  # Output: [B, L, N, D]


def nconv(x, A):
    """Efficient batch matrix multiplication for Graph adjacency."""
    if len(A.shape) == 2:
        return torch.einsum('ncvl,vw->ncwl', (x, A.to(x.device))).contiguous()
    else:
        return torch.einsum('ncvl,nvw->ncwl', (x, A.to(x.device))).contiguous()


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super().__init__()
        # Input channel expands based on 'order' (powers of adjacency matrix)
        self.order = order
        combined_in = (order * support_len + 1) * c_in
        self.final_conv = Conv2d(combined_in, c_out, (1, 1))
        self.bn = nn.BatchNorm2d(c_out)
        self.dropout = dropout

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = self.bn(h)
        return F.dropout(h, self.dropout, training=self.training)