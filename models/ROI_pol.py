"""ROI mean pooling over an AAL (or similar) atlas."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn


class ROIPooling3D(nn.Module):
    def __init__(
        self,
        atlas_path: str,
        roi_range: tuple = (1, 90),
    ) -> None:
        super().__init__()

        atlas = nib.load(atlas_path).get_fdata().astype(int)
        self.DHW = atlas.shape
        atlas_max = int(atlas.max())
        atlas_labels = sorted(int(x) for x in np.unique(atlas) if int(x) > 0)
        roi_start, roi_end = int(roi_range[0]), int(roi_range[1])
        roi_ids = np.arange(roi_start, roi_end + 1)
        missing = [int(r) for r in roi_ids if int(r) not in set(atlas_labels)]
        print(
            f"[ROIPooling3D] atlas={atlas_path} shape={self.DHW} "
            f"label_max={atlas_max} n_nonzero_labels={len(atlas_labels)} "
            f"roi_range=[{roi_start},{roi_end}] -> R={len(roi_ids)}"
        )
        if missing:
            print(
                f"[ROIPooling3D] WARNING: {len(missing)} ROI ids not present in atlas "
                f"(e.g. {missing[:5]}{'...' if len(missing) > 5 else ''}). "
                f"Those ROIs will pool as zeros / empty masks."
            )
        if atlas_max < roi_end:
            print(
                f"[ROIPooling3D] WARNING: atlas max label {atlas_max} < roi_end {roi_end}."
            )

        roi_index = np.zeros_like(atlas, dtype=np.int64)
        for out_idx, roi_id in enumerate(roi_ids, start=1):
            roi_index[atlas == int(roi_id)] = out_idx
        roi_index_t = torch.from_numpy(roi_index.reshape(-1)).long()
        counts = torch.bincount(roi_index_t, minlength=len(roi_ids) + 1)[1:].float()
        self.register_buffer("roi_index", roi_index_t)
        self.register_buffer("counts", counts.clamp_min(1.0))
        self.roi_range = (roi_start, roi_end)
        self.num_rois = int(len(roi_ids))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat [B,C,D,H,W] -> [B,R,C]."""
        d, h, w = self.DHW
        feat = feat[..., :d, :h, :w]

        b, c = feat.size(0), feat.size(1)
        r = self.num_rois

        feat_flat = feat.reshape(b, c, -1)
        roi_index = self.roi_index[: feat_flat.size(-1)]
        sums = torch.zeros(b, c, r + 1, device=feat.device, dtype=feat.dtype)
        scatter_index = roi_index.view(1, 1, -1).expand(b, c, -1)
        sums.scatter_add_(2, scatter_index, feat_flat)
        sums = sums[:, :, 1:]

        counts = self.counts[:r].to(device=feat.device, dtype=feat.dtype)
        return (sums / counts.view(1, 1, r)).transpose(1, 2)


class ROIClassifier(nn.Module):
    def __init__(self, r: int = 90, num_cls: int = 2, hidden: int = 512, feat_c: int = 64):
        super().__init__()
        in_dim = r * feat_c
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_cls),
        )

    def forward(self, roi_feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(roi_feat.flatten(1))
