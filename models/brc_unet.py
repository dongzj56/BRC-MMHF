"""Dual-stream 3D U-Net with bottleneck Channel Exchange (BRC image branch)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.channel_exchange import ChannelExchange
from models.unet3d import Conv3DBlock, UpConv3DBlock


class SharedBottleneckBlock(nn.Module):
    """Shared conv weights with modality-specific BN (for CEN gamma)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv3d(in_channels, mid, kernel_size=3, padding=1)
        self.mri_bn1 = nn.BatchNorm3d(mid)
        self.pet_bn1 = nn.BatchNorm3d(mid)
        self.conv2 = nn.Conv3d(mid, out_channels, kernel_size=3, padding=1)
        self.mri_bn2 = nn.BatchNorm3d(out_channels)
        self.pet_bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x_mri: torch.Tensor, x_pet: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, nn.Module, nn.Module]:
        m = self.relu(self.mri_bn1(self.conv1(x_mri)))
        m = self.relu(self.mri_bn2(self.conv2(m)))
        p = self.relu(self.pet_bn1(self.conv1(x_pet)))
        p = self.relu(self.pet_bn2(self.conv2(p)))
        return m, p, self.mri_bn2, self.pet_bn2


class BRCDualUNet3D(nn.Module):
    """MRI/PET dual-stream 3D U-Net with optional shared bottleneck + CEN.

    Outputs high-resolution feature maps (level_channels[0] channels) for ROI
    pooling, plus optional segmentation logits.
    """

    def __init__(
        self,
        in_channels: int = 1,
        level_channels=None,
        bottleneck_channel: int = 512,
        use_shared_layer: bool = True,
        use_channel_exchange: bool = True,
        exchange_ratio: float = 0.2,
        ema_momentum: float = 0.9,
        use_seg_task: bool = False,
        seg_classes: int = 1,
    ) -> None:
        super().__init__()
        if level_channels is None:
            level_channels = [64, 128, 256]
        l1, l2, l3 = level_channels
        self.level_channels = list(level_channels)
        self.use_shared_layer = bool(use_shared_layer)
        self.use_channel_exchange = bool(use_channel_exchange) and self.use_shared_layer
        self.use_seg_task = bool(use_seg_task)

        # Independent encoders
        self.mri_enc1 = Conv3DBlock(in_channels, l1)
        self.mri_enc2 = Conv3DBlock(l1, l2)
        self.mri_enc3 = Conv3DBlock(l2, l3)
        self.pet_enc1 = Conv3DBlock(in_channels, l1)
        self.pet_enc2 = Conv3DBlock(l1, l2)
        self.pet_enc3 = Conv3DBlock(l2, l3)

        if self.use_shared_layer:
            self.shared_bneck = SharedBottleneckBlock(l3, bottleneck_channel)
            self.cen = ChannelExchange(
                bottleneck_channel,
                exchange_ratio=exchange_ratio,
                ema_momentum=ema_momentum,
                use_ema=True,
            )
        else:
            self.mri_bneck = Conv3DBlock(l3, bottleneck_channel, bottleneck=True)
            self.pet_bneck = Conv3DBlock(l3, bottleneck_channel, bottleneck=True)
            self.shared_bneck = None
            self.cen = None

        # Independent decoders (feature maps, no class head inside UpConv)
        self.mri_dec3 = UpConv3DBlock(bottleneck_channel, res_channels=l3)
        self.mri_dec2 = UpConv3DBlock(l3, res_channels=l2)
        self.mri_dec1 = UpConv3DBlock(l2, res_channels=l1, last_layer=False)
        self.pet_dec3 = UpConv3DBlock(bottleneck_channel, res_channels=l3)
        self.pet_dec2 = UpConv3DBlock(l3, res_channels=l2)
        self.pet_dec1 = UpConv3DBlock(l2, res_channels=l1, last_layer=False)

        if self.use_seg_task:
            self.mri_seg_head = nn.Conv3d(l1, seg_classes, kernel_size=1)
            self.pet_seg_head = nn.Conv3d(l1, seg_classes, kernel_size=1)
        else:
            self.mri_seg_head = None
            self.pet_seg_head = None

    def _decode(
        self,
        bottleneck: torch.Tensor,
        residuals,
        dec3,
        dec2,
        dec1,
    ) -> torch.Tensor:
        x = dec3(bottleneck, residuals[2])
        x = dec2(x, residuals[1])
        x = dec1(x, residuals[0])
        return x

    def forward(
        self,
        mri: Optional[torch.Tensor] = None,
        pet: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if mri is None and pet is None:
            raise ValueError("At least one of mri/pet must be provided.")

        out: Dict[str, torch.Tensor] = {}

        # Single-modality path: independent U-Net without CEN
        if mri is not None and pet is None:
            x, r1 = self.mri_enc1(mri)
            x, r2 = self.mri_enc2(x)
            x, r3 = self.mri_enc3(x)
            if self.use_shared_layer:
                x, _, _, _ = self.shared_bneck(x, x)
            else:
                x, _ = self.mri_bneck(x)
            feat = self._decode(x, [r1, r2, r3], self.mri_dec3, self.mri_dec2, self.mri_dec1)
            out["mri_features"] = feat
            if self.mri_seg_head is not None:
                out["mri_seg"] = self.mri_seg_head(feat)
            return out

        if pet is not None and mri is None:
            x, r1 = self.pet_enc1(pet)
            x, r2 = self.pet_enc2(x)
            x, r3 = self.pet_enc3(x)
            if self.use_shared_layer:
                _, x, _, _ = self.shared_bneck(x, x)
            else:
                x, _ = self.pet_bneck(x)
            feat = self._decode(x, [r1, r2, r3], self.pet_dec3, self.pet_dec2, self.pet_dec1)
            out["pet_features"] = feat
            if self.pet_seg_head is not None:
                out["pet_seg"] = self.pet_seg_head(feat)
            return out

        # Dual-modality
        xm, rm1 = self.mri_enc1(mri)
        xm, rm2 = self.mri_enc2(xm)
        xm, rm3 = self.mri_enc3(xm)
        xp, rp1 = self.pet_enc1(pet)
        xp, rp2 = self.pet_enc2(xp)
        xp, rp3 = self.pet_enc3(xp)

        if self.use_shared_layer:
            xm, xp, bn_m, bn_p = self.shared_bneck(xm, xp)
            if self.use_channel_exchange:
                xm, xp = self.cen(xm, xp, bn_mri=bn_m, bn_pet=bn_p)
        else:
            xm, _ = self.mri_bneck(xm)
            xp, _ = self.pet_bneck(xp)

        fm = self._decode(xm, [rm1, rm2, rm3], self.mri_dec3, self.mri_dec2, self.mri_dec1)
        fp = self._decode(xp, [rp1, rp2, rp3], self.pet_dec3, self.pet_dec2, self.pet_dec1)
        out["mri_features"] = fm
        out["pet_features"] = fp
        if self.mri_seg_head is not None:
            out["mri_seg"] = self.mri_seg_head(fm)
        if self.pet_seg_head is not None:
            out["pet_seg"] = self.pet_seg_head(fp)
        return out
