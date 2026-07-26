"""Subject-specific multimodal hypergraph fusion with dynamic incidence attention and LANS."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_sim_matrix(x: torch.Tensor) -> torch.Tensor:
    """x: [N, C] -> [N, N] cosine similarity."""
    x = F.normalize(x, p=2, dim=-1, eps=1e-6)
    return x @ x.t()


def _large_negative(dtype: torch.dtype) -> float:
    if dtype.is_floating_point:
        return max(torch.finfo(dtype).min, -1e4)
    return -1e4


@torch.no_grad()
def build_subject_incidence(
    feat_mri: torch.Tensor,
    feat_pet: torch.Tensor,
    ks: Sequence[int] = (6, 18),
) -> torch.Tensor:
    """Build binary incidence H for one subject.

    Args:
        feat_mri / feat_pet: [N, C]
        ks: multi-scale k for intra-modal hyperedges

    Returns:
        H: [2N, E] float binary incidence
    """
    n = feat_mri.size(0)
    device = feat_mri.device
    dtype = feat_mri.dtype
    edges: List[torch.Tensor] = []

    for feat in (feat_mri, feat_pet):
        sim = _cosine_sim_matrix(feat)
        sim.fill_diagonal_(_large_negative(sim.dtype))
        for k in ks:
            kk = min(int(k), max(n - 1, 1))
            # for each ROI i, hyperedge = {i} U top-k neighbors
            _, nn_idx = torch.topk(sim, k=kk, dim=1)
            for i in range(n):
                members = torch.zeros(n, device=device, dtype=dtype)
                members[i] = 1.0
                members[nn_idx[i]] = 1.0
                edges.append(members)

    # inter-modal pairwise hyperedges: MRI_r -- PET_r
    for r in range(n):
        members = torch.zeros(2 * n, device=device, dtype=dtype)
        members[r] = 1.0
        members[n + r] = 1.0
        edges.append(members)

    # pack intra edges (currently length n each) into 2N rows
    # first 2 * n * len(ks) edges are intra (n*len(ks) MRI + n*len(ks) PET)
    n_intra = 2 * n * len(ks)
    h_cols = []
    for e_idx, e in enumerate(edges):
        col = torch.zeros(2 * n, device=device, dtype=dtype)
        if e_idx < n * len(ks):
            # MRI intra
            col[:n] = e
        elif e_idx < n_intra:
            # PET intra
            col[n:] = e
        else:
            col = e
        h_cols.append(col)
    return torch.stack(h_cols, dim=1)  # [2N, E]


@torch.no_grad()
def build_batch_incidence(
    feat_mri: torch.Tensor,
    feat_pet: torch.Tensor,
    ks: Sequence[int] = (6, 18),
) -> torch.Tensor:
    """Per-subject H stacked as [B, 2N, E]."""
    b = feat_mri.size(0)
    hs = [
        build_subject_incidence(feat_mri[i], feat_pet[i], ks=ks)
        for i in range(b)
    ]
    return torch.stack(hs, dim=0)


class DynamicIncidenceAttention(nn.Module):
    """Soft incidence H_hat from node/hyperedge features (masked to H>0)."""

    def __init__(self, in_dim: int, attn_dim: int = 64) -> None:
        super().__init__()
        self.node_proj = nn.Linear(in_dim, attn_dim, bias=False)
        self.edge_proj = nn.Linear(in_dim, attn_dim, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, V, C]
            h: [B, V, E] binary
        Returns:
            h_hat: [B, V, E]
        """
        # hyperedge features: mean of member nodes
        deg_e = h.sum(dim=1).clamp_min(1.0)  # [B, E]
        edge_feat = torch.einsum("bve,bvc->bec", h, x) / deg_e.unsqueeze(-1)

        q = self.node_proj(x)  # [B, V, A]
        k = self.edge_proj(edge_feat)  # [B, E, A]
        score = torch.einsum("bva,bea->bve", q, k) / (q.size(-1) ** 0.5)
        # mask non-members to -inf then softmax over nodes for each edge
        mask = h > 0
        score = score.masked_fill(~mask, _large_negative(score.dtype))
        # softmax over V for each hyperedge
        attn = torch.softmax(score, dim=1)
        attn = attn * mask.float()
        # renormalize in case of empty edges
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return attn


class HyperGraphConvLayer(nn.Module):
    """Node -> hyperedge -> node message passing (Eq.9-10 style)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x: [B, V, C], h: [B, V, E]
        """
        x = self.lin(x)
        # node degrees Dv, edge degrees De
        de = h.sum(dim=1).clamp_min(1.0)  # [B, E]
        dv = h.sum(dim=2).clamp_min(1.0)  # [B, V]
        # X_e = H^T X / De
        x_e = torch.einsum("bve,bvc->bec", h, x) / de.unsqueeze(-1)
        # X' = H X_e / Dv
        x_out = torch.einsum("bve,bec->bvc", h, x_e) / dv.unsqueeze(-1)
        return self.act(x_out)


class MultimodalHypergraphFusion(nn.Module):
    """Subject-specific multimodal hypergraph + optional H_hat + LANS."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 2,
        ks: Sequence[int] = (6, 18),
        use_hypergraph_conv: bool = True,
        use_hypergraph_attention: bool = True,
        attn_dim: int = 64,
    ) -> None:
        super().__init__()
        self.ks = tuple(int(k) for k in ks)
        self.use_hypergraph_conv = bool(use_hypergraph_conv)
        self.use_hypergraph_attention = bool(use_hypergraph_attention) and self.use_hypergraph_conv
        self.out_dim = int(out_dim)

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        layers = []
        for _ in range(num_layers):
            layers.append(HyperGraphConvLayer(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.attn = DynamicIncidenceAttention(hidden_dim, attn_dim=attn_dim) if self.use_hypergraph_attention else None

        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        # fallback when hypergraph conv disabled
        self.fallback = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def _lans_scores(self, h_hat: torch.Tensor) -> torch.Tensor:
        """beta_v = mean attention over incident hyperedges. [B, V]."""
        mask = h_hat > 0
        deg = mask.float().sum(dim=2).clamp_min(1.0)
        beta = (h_hat * mask.float()).sum(dim=2) / deg
        return beta

    def forward(
        self,
        feat_mri: torch.Tensor,
        feat_pet: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat_mri / feat_pet: [B, N, C]
        """
        b, n, c = feat_mri.shape
        if not self.use_hypergraph_conv:
            # mean pool each modality then MLP
            zm = feat_mri.mean(dim=1)
            zp = feat_pet.mean(dim=1)
            z = self.fallback(torch.cat([zm, zp], dim=-1))
            return {
                "node_features": torch.cat([feat_mri, feat_pet], dim=1),
                "image_feature": z,
                "lans_scores": None,
                "H": None,
                "H_hat": None,
            }

        x0 = torch.cat([feat_mri, feat_pet], dim=1)  # [B, 2N, C]
        x = self.input_proj(x0)
        h = build_batch_incidence(feat_mri, feat_pet, ks=self.ks)

        h_hat = None
        if self.use_hypergraph_attention:
            h_use = self.attn(x, h)
            h_hat = h_use
        else:
            h_use = h

        for layer in self.layers:
            x = layer(x, h_use)

        beta = self._lans_scores(h_hat if h_hat is not None else h_use)
        # attention-weighted pooling with LANS
        w = beta / beta.sum(dim=1, keepdim=True).clamp_min(1e-6)
        z = torch.einsum("bv,bvc->bc", w, x)
        z = self.pool_mlp(z)

        return {
            "node_features": x,
            "image_feature": z,
            "lans_scores": beta,
            "H": h,
            "H_hat": h_hat,
        }


class SingleModalityROIEncoder(nn.Module):
    """Mean-pool ROI features + MLP when only one image modality is used."""

    def __init__(self, in_dim: int, out_dim: int = 128, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, N, C]
        return self.net(feat.mean(dim=1))
