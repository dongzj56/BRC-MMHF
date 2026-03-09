
import torch
import torch.nn as nn
from models.unet3d import SharedConv3DBlock, Conv3DBlock, UpConv3DBlock

class TabularEncoder(nn.Module):
    """
    Lightweight MLP encoder for tabular data.
    Input: [B, input_dim]
    Output: [B, output_dim]
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class CEN_DualUNet3D(nn.Module):
    """
    Dual-stream 3D U-Net with Channel Exchange (CEN) in the encoder.
    Returns voxel-wise feature maps for both modalities (for ROI Pooling)
    and optional segmentation logits (for auxiliary loss).
    """
    def __init__(self,
                 in_channels=1,
                 num_classes=None, # If not None, adds a segmentation head
                 level_channels=[64, 128, 256],
                 bottleneck_channel=512,
                 share_layers=2,
                 cen_ratios=(0.2, 0.1)):
        super().__init__()
        
        self.share_layers = share_layers
        l1, l2, l3 = level_channels
        
        # --- Encoder (Shared / CEN) ---
        self.shared_blocks = nn.ModuleList()
        in_out_pairs = [(in_channels, l1), (l1, l2)]
        
        # Only support early sharing for now as per ImageEncoder_CEN logic
        for i in range(min(share_layers, 2)):
            ic, oc = in_out_pairs[i]
            ratio = cen_ratios[i] if i < len(cen_ratios) else 0.0
            self.shared_blocks.append(
                SharedConv3DBlock(ic, oc, half_ratio=ratio, with_pool=True)
            )
            
        # Remaining independent encoder blocks
        self.mri_block3 = Conv3DBlock(l2, l3)
        self.pet_block3 = Conv3DBlock(l2, l3)
        
        self.mri_bneck = Conv3DBlock(l3, bottleneck_channel, bottleneck=True)
        self.pet_bneck = Conv3DBlock(l3, bottleneck_channel, bottleneck=True)
        
        # --- Decoder (Independent) ---
        # We need decoders to reconstruct features for ROI pooling (high resolution)
        # Structure matches UNet3D_Feature
        
        # MRI Decoder
        self.mri_s3 = UpConv3DBlock(bottleneck_channel, res_channels=l3)
        self.mri_s2 = UpConv3DBlock(l3, res_channels=l2)
        self.mri_s1 = UpConv3DBlock(l2, res_channels=l1, last_layer=False)
        
        # PET Decoder
        self.pet_s3 = UpConv3DBlock(bottleneck_channel, res_channels=l3)
        self.pet_s2 = UpConv3DBlock(l3, res_channels=l2)
        self.pet_s1 = UpConv3DBlock(l2, res_channels=l1, last_layer=False)

        # Segmentation Heads (Auxiliary Task)
        self.num_classes = num_classes
        if num_classes is not None:
            # Output channels for segmentation (e.g., 2 for bg/fg, or more for AAL)
            # Assuming the features from s1 have l1 channels (64)
            self.mri_seg_head = nn.Conv3d(l1, num_classes, kernel_size=1)
            self.pet_seg_head = nn.Conv3d(l1, num_classes, kernel_size=1)


    def forward(self, mri, pet):
        # --- Encoder ---
        residuals_mri = []
        residuals_pet = []
        
        x_m = mri
        x_p = pet
        
        # Shared blocks (Layer 1, 2)
        for blk in self.shared_blocks:
            # SharedConv3DBlock returns: xm_p, xp_p, xm, xp
            # We need pooled for next layer, and unpooled (xm, xp) as residuals
            x_m, x_p, res_m, res_p = blk(x_m, x_p)
            residuals_mri.append(res_m)
            residuals_pet.append(res_p)
            
        x_m, res_m3 = self.mri_block3(x_m)
        x_p, res_p3 = self.pet_block3(x_p)
        
        residuals_mri.append(res_m3)
        residuals_pet.append(res_p3)

        x_m, _ = self.mri_bneck(x_m)
        x_p, _ = self.pet_bneck(x_p)
        
        # --- Decoder ---
        # s3 takes bottleneck + res3 (from block3)
        # s2 takes s3_out + res2 (from block2/shared2)
        # s1 takes s2_out + res1 (from block1/shared1)
        
        # Decoder logic needs to match encoder depth.
        # If share_layers=2, we have res from block0, block1.
        # Then res from block3.
        # residuals_mri list: [res0, res1, res3]
        
        # mri_s3 takes (x_m, residuals_mri[-1]) -> res3
        d_m = self.mri_s3(x_m, residuals_mri[-1])
        d_p = self.pet_s3(x_p, residuals_pet[-1])
        
        # mri_s2 takes (d_m, residuals_mri[-2]) -> res1 (from block1)
        d_m = self.mri_s2(d_m, residuals_mri[-2])
        d_p = self.pet_s2(d_p, residuals_pet[-2])
        
        # mri_s1 takes (d_m, residuals_mri[-3]) -> res0 (from block0)
        d_m = self.mri_s1(d_m, residuals_mri[-3])
        d_p = self.pet_s1(d_p, residuals_pet[-3])
        
        outputs = {
            "mri_features": d_m,
            "pet_features": d_p
        }
        
        if self.num_classes is not None:
            seg_m = self.mri_seg_head(d_m)
            seg_p = self.pet_seg_head(d_p)
            outputs["mri_seg"] = seg_m
            outputs["pet_seg"] = seg_p
            
        return outputs
