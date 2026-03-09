import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ROIPooling3D(nn.Module):
    def __init__(self,
                 atlas_path : str,
                 roi_range  : tuple = (1, 94)
                 ) -> None:
        super().__init__()

        atlas   = nib.load(atlas_path).get_fdata().astype(int)
        self.DHW = atlas.shape

        roi_ids   = np.arange(roi_range[0], roi_range[1] + 1)
        atlas_max = int(atlas.max())
        oh = F.one_hot(torch.from_numpy(atlas).long(),
                    num_classes=atlas_max + 1)[..., roi_ids]
        oh = oh.permute(3,0,1,2).float()
        self.register_buffer("onehot", oh)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        D, H, W = self.DHW
        feat = feat[..., :D, :H, :W]

        B, C = feat.size(0), feat.size(1)
        R    = self.onehot.size(0)

        feat_flat = feat.reshape(B, C, -1)
        mask_flat = self.onehot.reshape(R, -1)

        num = torch.einsum("rn,bcn->brc", mask_flat, feat_flat)

        den = mask_flat.sum(dim=1).clamp_min(1e-6)

        return num / den[None, :, None]

class ROIClassifier(nn.Module):
    def __init__(self, R: int = 94, num_cls: int = 2, hidden: int = 512):
        super().__init__()
        in_dim = R * 64
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_cls)
        )

    def forward(self, roi_feat: torch.Tensor) -> torch.Tensor:
        x = roi_feat.flatten(1)
        return self.mlp(x)
