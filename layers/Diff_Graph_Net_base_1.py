# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Conv2d, ModuleList
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Diff_Pooling(nn.Module):
    """
    Differential Pooling layer that selects key features based on signal gradients
    and refines them using a Graph Convolutional Network.
    """

    def __init__(self, num_nodes, in_dim, out_dim, d_model, d_ff, dropout=0.3, pooling='mean'):
        super().__init__()
        self.num_nodes = num_nodes
        self.gcn = GraphConvNet(in_dim, out_dim, dropout, support_len=1)
        self.num = 0

    def visual_attention(self, adj, save_path):
        """Visualizes the learned adjacency matrix (Attention) for pooling."""
        tmp = adj[0, :].clone()
        num_nodes = tmp.shape[-1]

        # Reorder nodes for better visualization if needed
        idx = torch.arange(num_nodes)
        idx = torch.cat((idx[0::2], idx[1::2]))
        tmp = tmp[idx, :][:, idx]

        plt.figure(figsize=(8, 6))
        sns.heatmap(tmp.detach().cpu().numpy(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Pooling Attention Matrix')
        plt.savefig(save_path)
        plt.close()

    def forward(self, x, d_x, max_len):
        # Find indices of maximum gradient magnitude
        x_max_index = torch.max(torch.abs(d_x), dim=-1, keepdim=True)[1]
        x_max_index_next = x_max_index + 1  # Neighboring point

        pooling_index = torch.cat([x_max_index, x_max_index_next], dim=-1)

        # Gather key points: [B, H, D, 2]
        pooling_value = torch.gather(x, dim=-1, index=pooling_index.long())
        B, H, D, _ = pooling_value.shape
        pooling_value = pooling_value.view(B, H, 2 * D)

        # Compute dynamic adjacency for pooled features
        adj = torch.matmul(pooling_value.transpose(-1, -2), pooling_value) / pooling_value.shape[-1]

        # Mask self-loops for softmax stabilization
        eye = torch.eye(2 * D, device=x.device).repeat(B, 1, 1)
        adj_masked = adj - (eye * 1e8)
        pooling_adj = torch.softmax(F.leaky_relu(adj_masked), dim=-1) + eye

        if not self.training and self.num < 1:
            self.num += 1
            self.visual_attention(pooling_adj, save_path='./adj_pooling.png')

        # Refine via GCN
        res = self.gcn(pooling_value.unsqueeze(-1), [pooling_adj]).squeeze(-1)
        res = res.view(B, H, D, 2).permute(0, 2, 3, 1).contiguous()
        res = res.view(B, D, 2 * H).transpose(-1, -2)

        return res


class TCN_base(nn.Module):
    """
    Temporal Convolutional Network with Graph Attention and Differential Pooling.
    """

    def __init__(self, num_nodes, in_dim, out_dim, d_model, d_ff,
                 dropout=0.3, kernel_size=3, blocks=2, layers=1, args=None):
        super().__init__()
        self.blocks = blocks
        self.layers = layers
        self.args = args

        self.start_conv = nn.Conv2d(in_dim, d_model, kernel_size=(1, 1))

        depth = blocks * layers
        self.diff_pooling = ModuleList([Diff_Pooling(num_nodes, d_model, d_model, d_model, d_ff) for _ in range(depth)])
        self.residual_convs = ModuleList([Conv2d(d_model, d_model, (1, 1)) for _ in range(depth)])
        self.skip_convs = ModuleList([Conv2d(d_model, d_ff, (1, 1)) for _ in range(depth)])
        self.bn = ModuleList([BatchNorm2d(d_model) for _ in range(depth)])
        self.graph_convs = ModuleList(
            [GraphConvNet(d_model, d_model, dropout, support_len=3, add_cross=True) for _ in range(depth)])
        self.diff_index_norm = ModuleList([nn.BatchNorm1d(d_model) for _ in range(depth)])

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()

        receptive_field = 1
        for b in range(blocks):
            dilation = 1
            for l in range(layers):
                padding = (kernel_size - 1) // 2
                self.filter_convs.append(
                    Conv2d(d_model, d_model, (1, kernel_size), dilation=dilation, padding=(0, padding)))
                self.gate_convs.append(
                    Conv2d(d_model, d_model, (1, kernel_size), dilation=dilation, padding=(0, padding)))
                receptive_field += (kernel_size - 1) * dilation
                dilation *= 2

        self.receptive_field = receptive_field
        self.output_bn = BatchNorm2d(d_model)

    def compute_diff(self, x):
        """Computes time-step differences with zero padding to maintain length."""
        d_x = torch.diff(x, n=1, dim=-1)
        padding = torch.zeros_like(x[:, :, :, -1:])
        d_x_padded = torch.cat((d_x, padding), dim=-1)
        return x, d_x_padded

    def forward(self, x, adj_list, **kwargs):
        x = self.start_conv(x)

        # Pad if input is shorter than receptive field
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))

        skip = 0
        for i in range(self.blocks * self.layers):
            residual = x
            i_x, d_x = self.compute_diff(x)

            # Differential Pooling
            pool_val = self.diff_pooling[i](x, d_x, d_x.shape[-1])

            # Dilated Gated Convolutions
            filt = torch.tanh(self.filter_convs[i](i_x))
            gate = torch.sigmoid(torch.abs(self.gate_convs[i](d_x)))
            x = filt * gate

            # Skip connection
            s = self.skip_convs[i](x)
            if isinstance(skip, torch.Tensor):
                skip = s + skip[:, :, :, -s.size(3):]
            else:
                skip = s

            # Dynamic Adjacency generation
            d_x_max, _ = torch.max(d_x, dim=-1)
            d_x_min, _ = torch.min(d_x, dim=-1)
            adj_feature = torch.where(d_x_max > -d_x_min, d_x_max, d_x_min)

            g1 = F.softmax(F.relu(torch.bmm(adj_feature.transpose(1, 2), adj_feature)), dim=-1)

            idx_feat = torch.max(torch.abs(d_x), dim=-1)[1].to(torch.float64)
            idx_feat = self.diff_index_norm[i](idx_feat)
            g2 = F.softmax(F.relu(torch.bmm(idx_feat.transpose(1, 2), idx_feat)), dim=-1)

            # Update dynamic supports (keep only original and 2 new ones)
            current_adj = [adj_list[0], g1, g2]

            if i < (self.blocks * self.layers - 1):
                x = self.graph_convs[i](x, current_adj, pool_val)
                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[i](x)

        out = self.output_bn(skip)
        return out.transpose(1, -1)


def nconv(x, A):
    """BMM-style multiplication for graph adjacency."""
    if len(A.shape) == 2:
        return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()
    else:
        return torch.einsum('ncvl,nvw->ncwl', (x, A)).contiguous()


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2, add_cross=False):
        super().__init__()
        self.order = order
        # Calculate concatenated input dimension
        inner_dim = (order * support_len + (3 if add_cross else 1)) * c_in
        self.final_conv = Conv2d(inner_dim, c_out, (1, 1))
        self.bn = nn.BatchNorm2d(c_out)
        self.dropout = dropout

    def forward(self, x, supports, cross_x=None):
        out = [x]
        for a in supports:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        if cross_x is not None:
            # Repeat pooling values across the temporal dimension
            cross_feat = cross_x.unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])
            out.append(cross_feat)

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = self.bn(h)
        return F.dropout(h, self.dropout, training=self.training)