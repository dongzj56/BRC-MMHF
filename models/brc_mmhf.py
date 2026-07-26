"""BRC-MMHF: dual-stream U-Net + multimodal hypergraph + tabular fusion."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.brc_unet import BRCDualUNet3D
from models.hypergraph import MultimodalHypergraphFusion, SingleModalityROIEncoder
from models.ROI_pol import ROIPooling3D
from models.tabular_encoder import TabularEncoder


class BRCMMHF(nn.Module):
    """Full multimodal classifier matching the BRC-MMHF paper pipeline."""

    def __init__(
        self,
        tab_dim: int = 133,
        num_classes: int = 2,
        aal_path: str = "",
        roi_range: Tuple[int, int] = (1, 90),
        level_channels=None,
        bottleneck_channel: int = 512,
        image_feature_dim: int = 128,
        tab_feature_dim: int = 64,
        roi_feat_dim: int = 64,
        use_mri: bool = True,
        use_pet: bool = True,
        use_table: bool = True,
        use_shared_layer: bool = True,
        use_channel_exchange: bool = True,
        use_hypergraph_conv: bool = True,
        use_hypergraph_attention: bool = True,
        use_seg_task: bool = False,
        ks: Sequence[int] = (6, 18),
        exchange_ratio: float = 0.2,
        dropout_rate: float = 0.5,
        hg_hidden_dim: int = 128,
        hg_num_layers: int = 2,
    ) -> None:
        super().__init__()
        if level_channels is None:
            level_channels = [64, 128, 256]

        self.use_mri = bool(use_mri)
        self.use_pet = bool(use_pet)
        self.use_table = bool(use_table)
        self.use_seg_task = bool(use_seg_task)
        self.image_feature_dim = int(image_feature_dim)
        self.tab_feature_dim = int(tab_feature_dim)
        self.roi_feat_dim = int(roi_feat_dim)

        need_image = self.use_mri or self.use_pet
        self.unet = None
        self.pool = None
        self.hg = None
        self.single_roi_enc = None

        if need_image:
            if not aal_path:
                raise ValueError("aal_path is required when using MRI/PET.")
            self.unet = BRCDualUNet3D(
                in_channels=1,
                level_channels=level_channels,
                bottleneck_channel=bottleneck_channel,
                use_shared_layer=use_shared_layer,
                use_channel_exchange=use_channel_exchange,
                exchange_ratio=exchange_ratio,
                use_seg_task=use_seg_task,
                seg_classes=1,
            )
            self.pool = ROIPooling3D(aal_path, roi_range=roi_range)
            if self.use_mri and self.use_pet:
                self.hg = MultimodalHypergraphFusion(
                    in_dim=roi_feat_dim,
                    hidden_dim=hg_hidden_dim,
                    out_dim=image_feature_dim,
                    num_layers=hg_num_layers,
                    ks=ks,
                    use_hypergraph_conv=use_hypergraph_conv,
                    use_hypergraph_attention=use_hypergraph_attention,
                )
            else:
                self.single_roi_enc = SingleModalityROIEncoder(
                    in_dim=roi_feat_dim,
                    out_dim=image_feature_dim,
                    hidden=hg_hidden_dim,
                )

        self.tab_encoder = None
        if self.use_table:
            self.tab_encoder = TabularEncoder(
                input_dim=int(tab_dim),
                output_dim=tab_feature_dim,
                hidden_dim=max(tab_feature_dim, 64),
            )

        clf_in = 0
        if need_image:
            clf_in += image_feature_dim
        if self.use_table:
            clf_in += tab_feature_dim
        if clf_in <= 0:
            raise ValueError("At least one modality must be enabled.")

        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        mri: Optional[torch.Tensor] = None,
        pet: Optional[torch.Tensor] = None,
        tabular: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        parts = []
        out: Dict[str, Optional[torch.Tensor]] = {
            "logits": None,
            "image_feature": None,
            "tabular_feature": None,
            "lans_scores": None,
            "mri_seg": None,
            "pet_seg": None,
        }

        if self.unet is not None:
            mri_in = mri if self.use_mri else None
            pet_in = pet if self.use_pet else None
            if self.use_mri and mri_in is None:
                raise ValueError("MRI is enabled but mri tensor is None.")
            if self.use_pet and pet_in is None:
                raise ValueError("PET is enabled but pet tensor is None.")

            img_out = self.unet(mri=mri_in, pet=pet_in)
            if "mri_seg" in img_out:
                out["mri_seg"] = img_out["mri_seg"]
            if "pet_seg" in img_out:
                out["pet_seg"] = img_out["pet_seg"]

            if self.use_mri and self.use_pet:
                rm = self.pool(img_out["mri_features"])
                rp = self.pool(img_out["pet_features"])
                hg_out = self.hg(rm, rp)
                z_i = hg_out["image_feature"]
                out["lans_scores"] = hg_out.get("lans_scores")
            elif self.use_mri:
                rm = self.pool(img_out["mri_features"])
                z_i = self.single_roi_enc(rm)
            else:
                rp = self.pool(img_out["pet_features"])
                z_i = self.single_roi_enc(rp)
            out["image_feature"] = z_i
            parts.append(z_i)

        if self.tab_encoder is not None:
            if tabular is None:
                raise ValueError("Table is enabled but tabular tensor is None.")
            z_t = self.tab_encoder(tabular)
            out["tabular_feature"] = z_t
            parts.append(z_t)

        fused = torch.cat(parts, dim=-1)
        out["logits"] = self.classifier(fused)
        return out


def build_brc_mmhf_from_cfg(cfg, tab_dim: int) -> BRCMMHF:
    """Helper used by MMHF.py / ablation scripts."""
    roi_start = int(getattr(cfg, "roi_start", 1))
    roi_end = int(getattr(cfg, "roi_end", 90))
    ks = getattr(cfg, "hg_ks", [6, 18])
    if isinstance(ks, dict):
        # legacy format {0: (6,18), 1: (6,18)}
        ks = list(ks.get(0, (6, 18)))
    return BRCMMHF(
        tab_dim=int(tab_dim),
        num_classes=int(getattr(cfg, "nb_class", 2)),
        aal_path=str(getattr(cfg, "AAL_dir", "")),
        roi_range=(roi_start, roi_end),
        level_channels=getattr(cfg, "level_channels", [64, 128, 256]),
        bottleneck_channel=int(getattr(cfg, "bottleneck_channel", 512)),
        image_feature_dim=int(getattr(cfg, "image_feature_dim", 128)),
        tab_feature_dim=int(getattr(cfg, "tab_feature_dim", 64)),
        use_mri=bool(getattr(cfg, "use_mri", True)),
        use_pet=bool(getattr(cfg, "use_pet", True)),
        use_table=bool(getattr(cfg, "use_table", True)),
        use_shared_layer=bool(getattr(cfg, "use_shared_layer", True)),
        use_channel_exchange=bool(getattr(cfg, "use_channel_exchange", True)),
        use_hypergraph_conv=bool(getattr(cfg, "use_hypergraph_conv", True)),
        use_hypergraph_attention=bool(getattr(cfg, "use_hypergraph_attention", True)),
        use_seg_task=bool(getattr(cfg, "use_seg_task", getattr(cfg, "seg_task", False))),
        ks=tuple(ks),
        exchange_ratio=float(getattr(cfg, "exchange_ratio", 0.2)),
        dropout_rate=float(getattr(cfg, "dropout_rate", 0.5)),
    )
