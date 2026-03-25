# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict


class GraphConvNet(nn.Module):
    """
    Spatial Graph Convolution Operation.
    Aggregates information across the learned dynamic adjacency matrices.
    """

    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.order = order
        # Calculate combined input dimension based on diffusion steps (order)
        combined_in = (order * support_len + 1) * c_in
        self.final_conv = nn.Conv2d(combined_in, c_out, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(c_out)
        self.dropout = dropout

    def forward(self, x, support: list):
        # x: [B, C, N, L]
        out = [x]
        for a in support:
            # Multi-hop diffusion
            x1 = torch.einsum('bcnl,bnw->bcwl', x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('bcnl,bnw->bcwl', x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)  # Concatenate along channel dimension
        h = self.final_conv(h)
        h = self.bn(h)
        return F.dropout(h, self.dropout, training=self.training)


class XLSTM_dynamic_graph(nn.Module):
    """
    Extended LSTM specialized for dynamic adjacency matrix (Graph) generation.
    Learns how sensor correlations evolve over time.
    """

    def __init__(self, in_feature, d_model, save_path, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.save_path = save_path

        # Gates
        self.input_gate = nn.Linear(in_feature, 1)
        self.q_k_activation = nn.Softplus()

        # Projections for Key and Value to form the "Now" correlation state
        self.key_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )
        self.value_proj = nn.Sequential(
            nn.Linear(in_feature, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )

        self.graph_norm = nn.LayerNorm(num_nodes)
        self.visual_flag = True
        self.flag = 0

    def visual_cell(self, adj, epoch_flag):
        """Heatmap visualization of the learned sensor relationships."""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        plt.figure(figsize=(8, 6))
        sns.heatmap(adj[0].detach().cpu().numpy(), annot=False, cmap='coolwarm', vmin=0, vmax=1)
        plt.title(f'Dynamic Graph State - Step {epoch_flag}')
        plt.savefig(os.path.join(self.save_path, f'cell_step_{epoch_flag}.png'))
        plt.close()

    def step(self, xt, cell_past):
        # xt: [B, N, C] -> Features per sensor at current patch
        # cell_past: [B, N, N] -> Previous graph state

        # 1. Compute Gating
        i_gate = torch.sigmoid(self.input_gate(xt))  # [B, N, 1]
        f_gate = 1 - i_gate  # Balanced forgetting/input for stability

        # 2. Key-Value Interaction (The "Graph Message")
        k = self.q_k_activation(self.key_proj(xt))
        v = self.q_k_activation(self.value_proj(xt))

        # Generate candidate adjacency: [B, N, N]
        candidate_adj = torch.matmul(k, v.transpose(-1, -2))
        candidate_adj = F.relu(self.graph_norm(candidate_adj))

        # 3. State Update (LSTM-style memory)
        # i_gate/f_gate broadcast across the [N, N] matrix
        new_cell = (f_gate * cell_past) + (i_gate * candidate_adj)

        return new_cell, new_cell

    def forward(self, x, cell_past=None, **kwargs):
        # x: [B, Patch_Num, Nodes, d_model]
        B, T, N, C = x.shape
        device = x.device

        if cell_past is None:
            cell_past = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)

        h_state = cell_past
        mode = kwargs.get('mode', 'train')

        for t in range(T):
            h_state, _ = self.step(x[:, t, :, :], h_state)

            # Visual log during testing
            if mode == 'test' and self.visual_flag:
                self.visual_cell(h_state, self.flag)
                self.flag += 1

        if mode == 'test':
            self.visual_flag = False  # Only visualize once per test run

        return h_state


class DiDA_Net(nn.Module):
    """
    Differential Dynamic Adjacency Network (DiDA-Net).
    Main model for RUL prediction using dynamic graphs.
    """

    def __init__(self, args, save_path):
        super().__init__()
        self.patch_len = args.patch_size
        self.patch_stride = args.patch_size
        self.d_model = args.d_model
        self.n_sensors = args.input_feature
        # Add 1 if auxiliary features (like Cycle ID) are appended
        self.idx_n_sensors = self.n_sensors + 1

        # Input projection
        self.input_map = nn.Sequential(
            nn.Linear(self.patch_len, self.d_model),
            nn.BatchNorm1d(self.d_model)
        )

        # Temporal backbone (Assume TCN_base is imported)
        # self.tcn = TCN_base(...) 

        # Dynamic Graph Learner
        self.xlstm_graph = XLSTM_dynamic_graph(
            in_feature=self.d_model,
            d_model=self.d_model,
            save_path=save_path,
            num_nodes=self.idx_n_sensors
        )

        # Regression Head
        self.regressor = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.idx_n_sensors ** 2, self.d_model)),
            ('relu1', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(self.d_model, 1))
        ]))

    def forward(self, x, **kwargs):
        # x: [B, L, D]
        idxs = kwargs.get('idx')
        device = x.device

        # Append Index/Cycle info to sensor data
        if idxs is not None:
            cycle_info = idxs.to(device).to(x.dtype)
            x = torch.cat((x, cycle_info), dim=-1)

        # 1. Patching: [B, L, N] -> [B, Num_Patches, N, Patch_Len]
        x_patched = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        B, T, N, P = x_patched.shape

        # 2. Mapping to Latent Space
        x_mapped = self.input_map(x_patched.reshape(-1, P))
        x_mapped = x_mapped.reshape(B, T, N, self.d_model)

        # 3. Dynamic Adjacency Learning via xLSTM
        # In this architecture, xLSTM receives mapped features and yields final graph state
        final_graph = self.xlstm_graph(x_mapped, mode=kwargs.get('mode'))

        # 4. Final RUL Prediction
        output = self.regressor(final_graph.reshape(B, -1))
        output = torch.clamp(output, min=0.0, max=1.0)  # Normalized RUL [0, 1]

        return None, output