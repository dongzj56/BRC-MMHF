"""Lightweight MLP encoder for tabular clinical features."""

from __future__ import annotations

import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    """Input [B, input_dim] -> [B, output_dim]."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
