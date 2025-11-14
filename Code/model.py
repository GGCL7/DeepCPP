import math
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from kan import KANLinear
    _HAS_KAN = True
except Exception:
    _HAS_KAN = False


class MLPBranch(nn.Module):
    def __init__(self, in_dim=773, out_dim=64, hidden=256, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)  # [B, 64]


try:
    from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
    from torch_geometric.utils import subgraph
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False

class GNNBranch(nn.Module):

    def __init__(self, node_in=48, edge_in=3,
                 hidden=128, out_dim=64,
                 dropout=0.2, pool_ratio=0.30):
        super().__init__()
        if not _HAS_PYG:
            raise RuntimeError("Need to install torch_geometric")
        self.hidden = hidden
        self.pool_ratio = float(pool_ratio)

        self.node_enc = nn.Linear(node_in, hidden)
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.conv1 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
        )
        self.bn1 = nn.BatchNorm1d(hidden)

        self.score_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        self.conv2 = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
        )
        self.bn2 = nn.BatchNorm1d(hidden)

        self.sub_readout_proj = nn.Linear(2 * hidden, hidden)
        self.back_proj = nn.Linear(hidden, hidden)
        self.back_gate = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

        self.readout = nn.Sequential(
            nn.Linear(4 * hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    @torch.no_grad()
    def _topk_per_graph(self, score: torch.Tensor, batch_ids: torch.Tensor, ratio: float):
        device = score.device
        num_graphs = int(batch_ids.max().item()) + 1
        idx_list = []
        for g in range(num_graphs):
            mask = (batch_ids == g)
            n_g = int(mask.sum().item())
            if n_g == 0:
                continue
            k_g = max(1, int(math.ceil(ratio * n_g)))
            s_g = score[mask]                              # [n_g]
            _, topk_loc = torch.topk(s_g, k_g, sorted=False)
            global_idx = mask.nonzero(as_tuple=False).view(-1)[topk_loc]
            idx_list.append(global_idx)
        if len(idx_list) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.sort(torch.cat(idx_list, dim=0))[0]

    def forward(self, batch):

        x = self.node_enc(batch.x)              # [N, h]
        e = self.edge_enc(batch.edge_attr)      # [E, h]


        h1 = self.conv1(x, batch.edge_index, e)  # [N, h]
        h1 = self.bn1(h1); h1 = F.relu(h1); h1 = self.dropout(h1)
        h1 = h1 + x


        scores = self.score_mlp(h1).squeeze(-1)   # [N]
        idx_keep = self._topk_per_graph(scores, batch.batch, self.pool_ratio)
        sub_edge_index, sub_e = subgraph(
            idx_keep, batch.edge_index, e, relabel_nodes=True, num_nodes=h1.size(0)
        )
        h1_sub = h1[idx_keep]                    # [K, h]
        sub_batch = batch.batch[idx_keep]        # [K]


        h2_sub = self.conv2(h1_sub, sub_edge_index, sub_e)  # [K, h]
        h2_sub = self.bn2(h2_sub); h2_sub = F.relu(h2_sub); h2_sub = self.dropout(h2_sub)
        h2_sub = h2_sub + h1_sub


        hs_mean = global_mean_pool(h2_sub, sub_batch)   # [B, h]
        hs_max  = global_max_pool(h2_sub, sub_batch)    # [B, h]
        hs_graph = torch.cat([hs_mean, hs_max], dim=-1) # [B, 2h]

        hs_graph_h = self.sub_readout_proj(hs_graph)    # [B, h]
        hs_broadcast = hs_graph_h[batch.batch]          # [N, h]

        gate = self.back_gate(torch.cat([h1, hs_broadcast], dim=-1))  # [N, h]
        h1_plus = h1 + gate * self.back_proj(hs_broadcast)            # [N, h]


        hg_mean = global_mean_pool(h1_plus, batch.batch)   # [B, h]
        hg_max  = global_max_pool(h1_plus, batch.batch)    # [B, h]
        hg_graph = torch.cat([hg_mean, hg_max], dim=-1)    # [B, 2h]

        fused = torch.cat([hg_graph, hs_graph], dim=-1)    # [B, 4h]
        out = self.readout(fused)                          # [B, out_dim=64]
        return out


class PairwiseCrossAttention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.scale = math.sqrt(dim)
        self.q_a = nn.Linear(dim, dim)
        self.k_a = nn.Linear(dim, dim)
        self.v_a = nn.Linear(dim, dim)
        self.q_b = nn.Linear(dim, dim)
        self.k_b = nn.Linear(dim, dim)
        self.v_b = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, a, b):

        Qa, Ka, Va = self.q_a(a), self.k_a(a), self.v_a(a)
        Qb, Kb, Vb = self.q_b(b), self.k_b(b), self.v_b(b)

        alpha_ab = torch.sigmoid((Qa * Kb).sum(-1, keepdim=True) / self.scale)  # [B,1]
        alpha_ba = torch.sigmoid((Qb * Ka).sum(-1, keepdim=True) / self.scale)  # [B,1]
        ctx_ab = alpha_ab * Vb
        ctx_ba = alpha_ba * Va
        fused = 0.5 * (ctx_ab + ctx_ba)
        return self.ln(0.5*(a + b) + fused)


class FusionPepNetDual(nn.Module):

    def __init__(self,
                 mlp_in_dim=773,
                 gnn_node_in=48,
                 gnn_edge_in=3,
                 branch_out=64,
                 use_kan=True):
        super().__init__()
        self.branch_mlp = MLPBranch(in_dim=mlp_in_dim, out_dim=branch_out)
        self.branch_gnn = GNNBranch(node_in=gnn_node_in, edge_in=gnn_edge_in, out_dim=branch_out)


        self.ca = PairwiseCrossAttention(dim=branch_out)


        self.fusion = nn.Sequential(
            nn.LayerNorm(branch_out * 3),
            nn.Dropout(0.2),
            nn.Linear(branch_out * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, branch_out)
        )


        self.use_kan = use_kan and _HAS_KAN
        if self.use_kan:
            self.classifier = KANLinear(
                in_features=branch_out, out_features=2,
                grid_size=5, spline_order=3,
                scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                enable_standalone_scale_spline=True,
                base_activation=nn.SiLU
            )
        else:
            self.classifier = nn.Linear(branch_out, 2)

    def forward(self, feat, graph_batch):

        f_mlp = self.branch_mlp(feat)          # [B,64]
        f_gnn = self.branch_gnn(graph_batch)   # [B,64]


        f_ca  = self.ca(f_mlp, f_gnn)          # [B,64]

        fused = torch.cat([f_mlp, f_gnn, f_ca], dim=-1)  # [B, 192]
        fused = self.fusion(fused)                        # [B, 64]

        logits = self.classifier(fused)                  # [B, 2]
        return logits, f_mlp, f_gnn, fused


def hsic_loss(X, Y):

    N = X.size(0)
    H = torch.eye(N, device=X.device) - (1.0 / N) * torch.ones((N, N), device=X.device)
    K = X @ X.t()
    R = Y @ Y.t()
    return torch.trace(K @ H @ R @ H) / (N * N)

def pair_hsic_loss(f_a, f_b, lambda_hsic=0.5):

    return lambda_hsic * hsic_loss(f_a, f_b)

