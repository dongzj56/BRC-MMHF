import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1) Multi-scale k-NN Hypergraph Construction (Dense H)
# ---------------------------------------------------------
class HyperGraphIncidenceBuilder:
    """
    ks = {0: (k_loc, k_glob), 1: (k_loc, k_glob)}
    """
    def __init__(self, ks):
        self.ks = ks            # dict[int, tuple]

    @torch.no_grad()
    def _build_modal_edges(self, feat, k_set):
        """
        Generate multi-scale hyperedges for single modality based on cosine similarity
        feat  : [N, C]
        k_set : (k_loc, k_glob)
        return: H_modal [N, E_modal]
        """
        N = feat.size(0)
        sim = F.normalize(feat, dim=1) @ F.normalize(feat, dim=1).T  # [N,N]
        H_list = []

        for k in k_set:                          # Build hyperedges for each k
            _, knn = sim.topk(k + 1, dim=-1)     # Includes self
            H = torch.zeros(N, N, device=feat.device)
            for e_idx, nbrs in enumerate(knn):   # N hyperedges
                H[nbrs, e_idx] = 1.
            H_list.append(H)

        return torch.cat(H_list, dim=1)          # [N, N*len(k_set)]

    @torch.no_grad()
    def __call__(self, feats):
        """
        feats : list[Tensor]  (len=2) each [N,C]
        return: H_dense [2N, E_total]
        """
        H_modal = []
        for midx, feat in enumerate(feats):
            H_modal.append(self._build_modal_edges(feat, self.ks[midx]))

        # —— Cross-modal binary hyperedges —— #
        N = feats[0].size(0)
        H_inter = torch.zeros(2 * N, N, device=feats[0].device)
        for r in range(N):
            H_inter[r, r]       = 1.   # mod1-node r
            H_inter[N + r, r]   = 1.   # mod2-node r

        # Concatenate: H = [H_mod1 | H_mod2 | H_inter]
        H = torch.cat([
            torch.block_diag(*H_modal),   # block diag to [2N, ΣE_m]
            H_inter                       # [2N, N]
        ], dim=1)
        return H                         # Dense matrix


# ---------------------------------------------------------
# 2) Dense HyperGraph Convolution
# ---------------------------------------------------------
class HyperGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, H):
        """
        x : [B, 2N, F]   H : [2N, E]
        """
        x = self.lin(x)                          # Linear projection
        DV = H.sum(dim=1).clamp(min=1.)          # Node degree [2N]
        DE = H.sum(dim=0).clamp(min=1.)          # Hyperedge degree [E]

        H_norm = (H / DV.sqrt().unsqueeze(1)) / DE.unsqueeze(0)
        A = H_norm @ H_norm.t()                  # [2N,2N]
        x = torch.bmm(A.expand(x.size(0), -1, -1), x)
        return F.relu(x)


# ---------------------------------------------------------
# 3) General Dual-Modal Hypergraph Fusion Module
# ---------------------------------------------------------
class DualModalHyperGraph(nn.Module):
    """
    Parameters
    ----
    in_dim      Input feature dimension C
    hidden_dim  Hypergraph convolution output dimension
    num_layers  Number of HGNNConv layers
    ks          {0:(k_loc,k_glob), 1:(k_loc,k_glob)}
    """
    def __init__(self,
                 in_dim=64,
                 hidden_dim=128,
                 num_layers=2,
                 ks={0: (6, 18), 1: (4, 12)}):
        super().__init__()
        self.builder = HyperGraphIncidenceBuilder(ks)
        self.layers = nn.ModuleList([
            HyperGraphConv(in_dim if l == 0 else hidden_dim, hidden_dim)
            for l in range(num_layers)
        ])

    def forward(self, feat_mod1, feat_mod2):
        """
        feat_mod1 / feat_mod2 : [B, N, C]
        """
        B, N, _ = feat_mod1.shape
        H = self.builder([feat_mod1.mean(0), feat_mod2.mean(0)])   # [2N,E]

        x = torch.cat([feat_mod1, feat_mod2], dim=1)               # [B,2N,C]
        for layer in self.layers:
            x = layer(x, H)

        out_mod1, out_mod2 = x[:, :N, :], x[:, N:, :]
        return out_mod1, out_mod2            # [B,N,hidden_dim] × 2

class CrossModalEdgeAttention(nn.Module):
    """
    Perform QK^T/√d → softmax → V within 1-to-1 binary hyperedge (mod1_r, mod2_r)
    Input:
        x1, x2 : [B, N, H] (node embeddings of two modalities)
    Output:
        z1, z2 : [B, N, H] (updated nodes after attention),  α : [B, N] (weights)
    """
    def __init__(self, dim, dk=64):
        super().__init__()
        self.Wq = nn.Linear(dim, dk, bias=False)
        self.Wk = nn.Linear(dim, dk, bias=False)
        self.Wv = nn.Linear(dim, dk, bias=False)   # Can set dk=dim
        self.scale = dk ** -0.5

    def forward(self, x1, x2):
        # Q from mod-1, K/V from mod-2
        Q = self.Wq(x1)                     # [B,N,dk]
        K = self.Wk(x2)
        V = self.Wv(x2)

        # 1-to-1 attention: essentially scalar per brain region
        score = (Q * K).sum(dim=-1) * self.scale   # [B,N]
        α12   = torch.softmax(score, dim=-1)       # [B,N]

        # Update nodes
        z1 = α12.unsqueeze(-1) * V                 # [B,N,dk]

        # Swap order and calculate again for z2
        score21 = (self.Wq(x2) * self.Wk(x1)).sum(dim=-1) * self.scale
        α21     = torch.softmax(score21, dim=-1)
        z2 = α21.unsqueeze(-1) * self.Wv(x1)

        return z1, z2, α12, α21

class DualModalHyperGraphWithAttn(nn.Module):
    def __init__(self,
                 in_dim=64,
                 hidden_dim=128,
                 num_layers=2,
                 ks={0:(6,18),1:(4,12)},
                 dk=64):
        super().__init__()
        self.hg = DualModalHyperGraph(in_dim, hidden_dim, num_layers, ks)
        self.attn = CrossModalEdgeAttention(hidden_dim, dk)

        # Aggregate to global vector
        self.pool = nn.Linear(dk, dk)      # Simple linear + tanh
        self.act  = nn.Tanh()

    def forward(self, feat1, feat2):
        o1, o2 = self.hg(feat1, feat2)     # [B,N,H]

        z1, z2, α12, α21 = self.attn(o1, o2)   # [B,N,dk], [B,N]

        # Global aggregation: ∑ α_i · z_i
        g1 = (α12.unsqueeze(-1) * z1).sum(dim=1)   # [B,dk]
        g2 = (α21.unsqueeze(-1) * z2).sum(dim=1)   # [B,dk]

        g1 = self.act(self.pool(g1))               # Non-linear
        g2 = self.act(self.pool(g2))

        return {
            "node_mod1": z1, "node_mod2": z2,      # Updated node features
            "global_mod1": g1, "global_mod2": g2,  # Fused global vectors
            "alpha_m1": α12, "alpha_m2": α21       # Attention weights
        }
