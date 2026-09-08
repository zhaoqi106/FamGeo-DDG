import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter_max

from config import ESM_REDUCED_DIM


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf=16, d_min=0.0, d_max=20.0):
        super().__init__()
        self.num_rbf = int(num_rbf)
        self.d_min = float(d_min)
        self.d_max = float(d_max)

        centers = torch.linspace(self.d_min, self.d_max, self.num_rbf)
        gamma = 1.0 / ((self.d_max - self.d_min) / self.num_rbf) ** 2

        self.register_buffer("centers", centers)
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def forward(self, edge_index, pos):
        pos = torch.nan_to_num(pos, nan=0.0, posinf=1e4, neginf=-1e4)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        dist = torch.clamp(dist, min=0.0, max=100.0)

        diff = dist.view(-1, 1) - self.centers.view(1, -1)
        exp_input = -self.gamma * diff ** 2
        exp_input = torch.clamp(exp_input, min=-50.0, max=50.0)

        rbf = torch.exp(exp_input)
        rbf = torch.nan_to_num(rbf, nan=0.0, posinf=1.0, neginf=0.0)
        return rbf, dist


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3, use_structure_attention=False):
        super().__init__()
        self.use_structure_attention = bool(use_structure_attention)

        self.msg_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        self.norm = LayerNorm(hidden_dim)

        if self.use_structure_attention:
            self.attn_query = nn.Linear(hidden_dim, hidden_dim)
            self.attn_key = nn.Linear(hidden_dim, hidden_dim)
            self.attn_scale = math.sqrt(hidden_dim)

    def forward(self, h, pos, edge_index, edge_attr):
        row, col = edge_index

        h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
        pos = torch.nan_to_num(pos, nan=0.0, posinf=1e4, neginf=-1e4)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1e4, neginf=-1e4)

        m_ij = torch.cat([h[row], h[col], edge_attr], dim=-1)
        m_ij = self.msg_mlp(m_ij)

        diff = pos[row] - pos[col]
        dist_vec = diff.norm(dim=-1, keepdim=True) + 1e-8

        if self.use_structure_attention:
            q = self.attn_query(h[row]) / self.attn_scale
            k = self.attn_key(h[col])
            score = (q * k).sum(dim=-1) / dist_vec.squeeze(-1)
            attn = softmax(score, index=row)
            attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)
            m_ij = m_ij * attn.unsqueeze(-1)

        msg_aggr = scatter_add(m_ij, row, dim=0, dim_size=h.size(0))
        msg_aggr = torch.nan_to_num(msg_aggr, nan=0.0, posinf=1e4, neginf=-1e4)

        h = h + self.node_update(msg_aggr)
        h = self.norm(h)
        h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)

        delta_pos = diff / dist_vec
        coord_scalar = self.coord_mlp(m_ij).squeeze(-1)
        coord_delta = scatter_add(coord_scalar.unsqueeze(-1) * delta_pos, row, dim=0, dim_size=pos.size(0))
        coord_delta = torch.nan_to_num(coord_delta, nan=0.0, posinf=1e4, neginf=-1e4)

        pos = pos + coord_delta
        pos = torch.nan_to_num(pos, nan=0.0, posinf=1e4, neginf=-1e4)

        return h, pos


# Node input: Base(41)+DSSP(14)+DeltaESM(2560)
# Readout: Global Mean/Max + Mutation-local Mean/Max; optionally concat local DeltaESM graph representation
class LocalMutationGNN(nn.Module):
    def __init__(
        self,
        node_in_dim,
        hidden_dim,
        num_layers,
        dropout,
        esm_dim,
        use_esm_in_model=True,
        esm_hops=1,
        esm_scale=0.1,
        esm_pool_scope="mut_hops",
        esm_pool_stat="mean",
        esm_fuse_mode="concat",
        use_mutation_local_pooling=True,
        local_pool_fallback="global",
        use_structure_attention=False,
    ):
        super().__init__()

        self.use_esm_in_model = bool(use_esm_in_model)
        self.esm_hops = int(esm_hops)
        self.esm_pool_scope = str(esm_pool_scope).lower()
        self.esm_pool_stat = str(esm_pool_stat).lower()
        self.esm_fuse_mode = str(esm_fuse_mode).lower()
        self.use_mutation_local_pooling = bool(use_mutation_local_pooling)
        self.local_pool_fallback = str(local_pool_fallback).lower()
        self.use_structure_attention = bool(use_structure_attention)

        if self.use_esm_in_model:
            self.esm_reducer = nn.Linear(esm_dim, ESM_REDUCED_DIM)
            self.esm_graph_proj = nn.Linear(ESM_REDUCED_DIM, hidden_dim)
            self.esm_dropout = nn.Dropout(dropout)
            self.esm_scale = nn.Parameter(torch.tensor(float(esm_scale), dtype=torch.float32))
            self.esm_dim_reduced = ESM_REDUCED_DIM
            self.basic_dim = int(node_in_dim - esm_dim)
        else:
            self.esm_dim_reduced = 0
            self.basic_dim = int(node_in_dim)

        self.basic_proj = nn.Linear(self.basic_dim, hidden_dim)
        self.basic_dropout = nn.Dropout(dropout)

        self.node_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            LayerNorm(hidden_dim),
        )

        self.rbf = GaussianRBF(num_rbf=16, d_min=0.0, d_max=20.0)
        self.edge_embed = nn.Linear(16, hidden_dim)

        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_structure_attention=self.use_structure_attention,
                )
                for _ in range(int(num_layers))
            ]
        )

        esm_readout_extra = 0
        if self.use_esm_in_model and self.esm_fuse_mode == "concat":
            if self.esm_pool_stat in ["mean+max", "mean_max", "meanmax"]:
                esm_readout_extra = 2 * hidden_dim
            else:
                esm_readout_extra = hidden_dim

        self.readout_in_dim = 4 * hidden_dim + esm_readout_extra

        self.readout_mlp = nn.Sequential(
            nn.Linear(self.readout_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _pool_esm_graph(self, esm_reduced, local_mask, batch, num_graphs):
        device = esm_reduced.device
        dtype = esm_reduced.dtype
        dim = esm_reduced.size(-1)

        if local_mask is None or (not local_mask.any()):
            g_mean = torch.zeros((num_graphs, dim), device=device, dtype=dtype)
            g_max = torch.zeros((num_graphs, dim), device=device, dtype=dtype)
            return g_mean, g_max

        local_batch = batch[local_mask]
        local_esm = esm_reduced[local_mask]

        g_sum = scatter_add(local_esm, local_batch, dim=0, dim_size=num_graphs)
        cnt = scatter_add(
            torch.ones((local_esm.size(0), 1), device=device, dtype=dtype),
            local_batch,
            dim=0,
            dim_size=num_graphs,
        )
        g_mean = g_sum / (cnt + 1e-8)

        g_max, _ = scatter_max(local_esm, local_batch, dim=0, dim_size=num_graphs)
        g_max = torch.nan_to_num(g_max, nan=0.0, posinf=0.0, neginf=0.0)
        return g_mean, g_max

    def _get_mut_mask(self, x_basic):
        if x_basic.size(1) < 41:
            return None
        mut_flag = x_basic[:, 40]
        return mut_flag > 0.5

    def _expand_mut_local_mask(self, mut_mask, edge_index):
        if mut_mask is None or (not mut_mask.any()):
            return None

        if self.esm_pool_scope == "mut_site":
            return mut_mask.clone()

        local_mask = mut_mask.clone()
        if self.esm_hops <= 0:
            return local_mask

        row, col = edge_index
        for _ in range(self.esm_hops):
            new_mask = local_mask.clone()
            new_mask[col[local_mask[row]]] = True
            new_mask[row[local_mask[col]]] = True
            if torch.equal(new_mask, local_mask):
                break
            local_mask = new_mask

        return local_mask

    def forward(self, data):
        model_device = next(self.parameters()).device

        x = data.x.to(model_device, non_blocking=True)
        pos = data.pos.to(model_device, non_blocking=True)
        edge_index = data.edge_index.to(model_device, non_blocking=True)

        batch = data.batch.to(model_device, non_blocking=True) if getattr(data, "batch", None) is not None else None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=model_device)
        else:
            batch = batch.long()

        if hasattr(data, "num_graphs") and data.num_graphs is not None:
            num_graphs = int(data.num_graphs)
        elif hasattr(data, "ptr") and data.ptr is not None:
            num_graphs = int(data.ptr.numel() - 1)
        else:
            num_graphs = int(batch.max().item()) + 1

        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        pos = torch.nan_to_num(pos, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.use_esm_in_model:
            x_basic = x[:, : self.basic_dim]
            x_esm = x[:, self.basic_dim:]

            x_esm_red = self.esm_reducer(x_esm)
            x_esm_red = F.relu(x_esm_red)

            h = self.basic_proj(x_basic)
            h = self.basic_dropout(h)
            h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)

            mut_mask = self._get_mut_mask(x_basic)
            local_mask = self._expand_mut_local_mask(mut_mask, edge_index)

            g_esm = h.new_zeros((num_graphs, self.esm_dim_reduced))
            if local_mask is not None and local_mask.any():
                g_mean, g_max = self._pool_esm_graph(x_esm_red, local_mask, batch, num_graphs)
                if self.esm_pool_stat in ["mean+max", "mean_max", "meanmax"]:
                    g_esm = torch.cat([g_mean, g_max], dim=-1)
                else:
                    g_esm = g_mean

            if self.esm_pool_stat in ["mean+max", "mean_max", "meanmax"]:
                g_mean = g_esm[:, : self.esm_dim_reduced]
                g_max = g_esm[:, self.esm_dim_reduced:]
                g_esm_h = torch.cat(
                    [self.esm_graph_proj(g_mean), self.esm_graph_proj(g_max)],
                    dim=-1,
                )
            else:
                g_esm_h = self.esm_graph_proj(g_esm)

            g_esm_h = self.esm_dropout(g_esm_h)
            g_esm_h = self.esm_scale * g_esm_h
            g_esm_h = torch.nan_to_num(g_esm_h, nan=0.0, posinf=1e4, neginf=-1e4)
        else:
            x_basic = x[:, : self.basic_dim]
            h = self.basic_proj(x_basic)
            h = self.basic_dropout(h)
            h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
            mut_mask = self._get_mut_mask(x_basic)
            g_esm_h = None

        h = self.node_embedding(h)
        h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)

        rbf_feat, _ = self.rbf(edge_index, pos)
        edge_attr = self.edge_embed(rbf_feat)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1e4, neginf=-1e4)

        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)

        global_mean = global_mean_pool(h, batch)
        global_max = global_max_pool(h, batch)

        if self.use_mutation_local_pooling and (mut_mask is not None) and mut_mask.any():
            mut_h = h[mut_mask]
            mut_batch = batch[mut_mask]

            local_mean = h.new_zeros(global_mean.shape)
            local_max = h.new_zeros(global_max.shape)

            local_mean_valid = global_mean_pool(mut_h, mut_batch)
            local_max_valid = global_max_pool(mut_h, mut_batch)
            graph_ids = torch.unique(mut_batch)

            local_mean[graph_ids] = local_mean_valid
            local_max[graph_ids] = local_max_valid
        else:
            if self.local_pool_fallback == "zero":
                local_mean = h.new_zeros(global_mean.shape)
                local_max = h.new_zeros(global_max.shape)
            else:
                local_mean = global_mean
                local_max = global_max

        graph_repr = torch.cat([global_mean, global_max, local_mean, local_max], dim=-1)

        if self.use_esm_in_model and self.esm_fuse_mode == "concat" and g_esm_h is not None:
            graph_repr = torch.cat([graph_repr, g_esm_h], dim=-1)

        graph_repr = torch.nan_to_num(graph_repr, nan=0.0, posinf=1e4, neginf=-1e4)

        reg_out = self.readout_mlp(graph_repr).view(-1)
        reg_out = torch.nan_to_num(reg_out, nan=0.0, posinf=1e4, neginf=-1e4)
        reg_out = torch.clamp(reg_out, min=-1e2, max=1e2)
        return reg_out


# Huber + alpha * (1 - Pearson)
def pearson_correlation(pred, target, eps=1e-8):
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    cov = torch.mean(pred_centered * target_centered)
    pred_std = torch.sqrt(torch.mean(pred_centered ** 2) + eps)
    target_std = torch.sqrt(torch.mean(target_centered ** 2) + eps)

    pearson = cov / (pred_std * target_std + eps)
    pearson = torch.clamp(pearson, min=-1.0, max=1.0)
    return pearson


def mixed_ddg_loss(pred, target, alpha=0.1):
    huber_criterion = nn.SmoothL1Loss()
    huber_loss = huber_criterion(pred, target)

    if pred.size(0) >= 2:
        pearson = pearson_correlation(pred, target, eps=1e-8)
        pearson_loss = 1.0 - pearson
        if not torch.isfinite(pearson_loss):
            pearson_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    else:
        pearson_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    total_loss = huber_loss + float(alpha) * pearson_loss
    return total_loss, huber_loss, pearson_loss
