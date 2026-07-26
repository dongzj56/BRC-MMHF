"""Channel Exchange Network (CEN): swap weak channels ranked by |BN gamma|."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ChannelExchange(nn.Module):
    """Exchange the weakest floor(p*C) channels between MRI and PET.

    Channel importance is |BN.weight| (gamma). Optional EMA over training.
    No trainable parameters.
    """

    def __init__(
        self,
        num_channels: int,
        exchange_ratio: float = 0.2,
        ema_momentum: float = 0.9,
        use_ema: bool = True,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.exchange_ratio = float(exchange_ratio)
        self.ema_momentum = float(ema_momentum)
        self.use_ema = bool(use_ema)
        self.register_buffer("mri_score", torch.zeros(num_channels), persistent=True)
        self.register_buffer("pet_score", torch.zeros(num_channels), persistent=True)
        self.register_buffer("ema_initialized", torch.tensor(0, dtype=torch.uint8), persistent=True)

    @staticmethod
    def _gamma_from(
        bn: Optional[nn.Module] = None,
        gamma: Optional[torch.Tensor] = None,
        num_channels: int = 0,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        if gamma is not None:
            g = gamma.detach().abs().flatten()
        elif bn is not None and getattr(bn, "weight", None) is not None:
            g = bn.weight.detach().abs().flatten()
        else:
            g = torch.ones(num_channels, device=device, dtype=dtype)
        return g

    def _update_ema(self, mri_g: torch.Tensor, pet_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_ema:
            return mri_g, pet_g
        if int(self.ema_initialized.item()) == 0:
            self.mri_score.copy_(mri_g)
            self.pet_score.copy_(pet_g)
            self.ema_initialized.fill_(1)
        else:
            m = self.ema_momentum
            self.mri_score.mul_(m).add_(mri_g, alpha=1.0 - m)
            self.pet_score.mul_(m).add_(pet_g, alpha=1.0 - m)
        return self.mri_score, self.pet_score

    def forward(
        self,
        feat_mri: torch.Tensor,
        feat_pet: torch.Tensor,
        bn_mri: Optional[nn.Module] = None,
        bn_pet: Optional[nn.Module] = None,
        gamma_mri: Optional[torch.Tensor] = None,
        gamma_pet: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rank-wise one-to-one swap of weakest channels.

        Args:
            feat_mri / feat_pet: [B, C, D, H, W]
        """
        b, c, _, _, _ = feat_mri.shape
        k = int(math.floor(self.exchange_ratio * c))
        if k <= 0:
            return feat_mri, feat_pet

        mri_g = self._gamma_from(bn_mri, gamma_mri, c, feat_mri.device, feat_mri.dtype)
        pet_g = self._gamma_from(bn_pet, gamma_pet, c, feat_pet.device, feat_pet.dtype)
        if self.training and self.use_ema:
            mri_score, pet_score = self._update_ema(mri_g, pet_g)
        else:
            if self.use_ema and int(self.ema_initialized.item()) == 1:
                mri_score, pet_score = self.mri_score, self.pet_score
            else:
                mri_score, pet_score = mri_g, pet_g

        # ascending: weakest first
        mri_idx = torch.argsort(mri_score)[:k]
        pet_idx = torch.argsort(pet_score)[:k]

        out_m = feat_mri.clone()
        out_p = feat_pet.clone()
        # rank-wise swap
        tmp = out_m[:, mri_idx].clone()
        out_m[:, mri_idx] = out_p[:, pet_idx]
        out_p[:, pet_idx] = tmp
        return out_m, out_p
