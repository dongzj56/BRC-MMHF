import torch
from torch import nn
from models.unet3d import SharedConv3DBlock, Conv3DBlock, cen_exchange, UNet3DEncoder


class ImageEncoder(nn.Module):
    """
    Two-stream 3D-U-Net encoder, intra-stream independent learning -> feature concatenation, returns fused vector representation.
    """
    def __init__(
        self,
        in_channels_per_modality: int = 1,
        level_channels: list[int]  = [64, 128, 256],
        bottleneck_channel: int    = 512
    ):
        super().__init__()
        # Two independent Encoders (parameters not shared)
        self.mri_enc = UNet3DEncoder(
            in_channels_per_modality,
            level_channels,
            bottleneck_channel
        )
        self.pet_enc = UNet3DEncoder(
            in_channels_per_modality,
            level_channels,
            bottleneck_channel
        )
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, mri: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        # Intra-stream encoding
        f_mri = self.mri_enc(mri)  # [B, C, D, H, W]
        f_pet = self.pet_enc(pet)
        # Intra-stream pooling & flattening
        v_mri = self.global_pool(f_mri).flatten(1)  # [B, C]
        v_pet = self.global_pool(f_pet).flatten(1)  # [B, C]
        # Feature fusion
        # fused = torch.cat([v_mri, v_pet], dim=1)   # [B, 2C]
        # return fused
        return v_mri, v_pet

class ImageEncoder_CEN(nn.Module):
    """
    Dual-stream 3D-U-Net + CEN shared encoder, returns image feature vectors (excluding concatenation and classification head).
    """
    def __init__(
        self,
        in_ch_modality: int = 1,
        level_channels: list[int] = [64, 128, 256],
        bottleneck_ch: int = 512,
        share_layers: int = 2,
        cen_ratios: tuple[float, ...] = (0.2, 0.1),
        share_scheme: str = "early"
    ):
        super().__init__()
        assert share_scheme in ("early", "bottleneck")
        if share_scheme == "early":
            assert share_layers == len(cen_ratios)

        l1, l2, l3 = level_channels

        self.share_scheme = share_scheme
        self.shared_blocks = nn.ModuleList()
        self.shared_bneck = None
        
        # Define independent early layers if not sharing early
        self.mri_block1 = None
        self.mri_block2 = None
        self.pet_block1 = None
        self.pet_block2 = None

        if share_scheme == "early":
            in_out_pairs = [
                (in_ch_modality, l1),
                (l1, l2),
                (l2, l3)
            ]
            for i in range(share_layers):
                ic, oc = in_out_pairs[i]
                ratio = cen_ratios[i]
                self.shared_blocks.append(
                    SharedConv3DBlock(ic, oc, half_ratio=ratio)
                )
            last_out = [l1, l2, l3][share_layers - 1]
        else:
            # Independent early layers
            self.mri_block1 = Conv3DBlock(in_ch_modality, l1)
            self.mri_block2 = Conv3DBlock(l1, l2)
            self.pet_block1 = Conv3DBlock(in_ch_modality, l1)
            self.pet_block2 = Conv3DBlock(l1, l2)
            
            last_out = l2
            b_ratio = cen_ratios[-1] if len(cen_ratios) > 0 else 0.2
            self.shared_bneck = SharedConv3DBlock(l3, bottleneck_ch, half_ratio=b_ratio, with_pool=False)

        self.mri_block3 = Conv3DBlock(last_out, l3)
        self.pet_block3 = Conv3DBlock(last_out, l3)
        if share_scheme == "early":
            self.mri_bneck = Conv3DBlock(l3, bottleneck_ch, bottleneck=True)
            self.pet_bneck = Conv3DBlock(l3, bottleneck_ch, bottleneck=True)

        # --- Global pooling ---
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, mri: torch.Tensor, pet: torch.Tensor):
        if self.share_scheme == "early":
            for blk in self.shared_blocks:
                mri, pet, _, _ = blk(mri, pet)
            mri, _ = self.mri_block3(mri)
            pet, _ = self.pet_block3(pet)
            mri, _ = self.mri_bneck(mri)
            pet, _ = self.pet_bneck(pet)
        else:
            # Process through independent early layers
            mri, _ = self.mri_block1(mri)
            mri, _ = self.mri_block2(mri)
            
            pet, _ = self.pet_block1(pet)
            pet, _ = self.pet_block2(pet)
            
            mri, _ = self.mri_block3(mri)
            pet, _ = self.pet_block3(pet)
            mri, pet, _, _ = self.shared_bneck(mri, pet)

        # ----- Feature vector extraction -----
        vm = self.gap(mri).flatten(1)  # [B, bottleneck_ch]
        vp = self.gap(pet).flatten(1)  # [B, bottleneck_ch]
        # Concatenate MRI and PET vectors        ↓ Dim [B, 2*bottleneck_ch]
        v = torch.cat([vm, vp], dim=1)

        return v
        #
        # # Return MRI and PET feature vectors
        # return vm, vp

#
# class MultiModalClassifier(nn.Module):
#     """
#     Multimodal fusion classifier: accepts MRI features, PET features and tabular features,
#     concatenates them and performs linear classification.
#
#     Inputs:
#       - vm: [B, C] MRI feature vector
#       - vp: [B, C] PET feature vector
#       - tab: [B, T] Tabular feature vector
#     Outputs:
#       - logits: [B, num_classes]
#     """
#     def __init__(
#         self,
#         feature_dim: int,
#         tabular_dim: int,
#         num_classes: int,
#         dropout: float = 0.0
#     ):
#         super().__init__()
#         fused_dim = feature_dim * 2 + tabular_dim
#         layers = []
#         if dropout > 0:
#             layers.append(nn.Dropout(dropout))
#         layers.append(nn.Linear(fused_dim, num_classes))
#         self.classifier = nn.Sequential(*layers)
#
#     def forward(self, vm: torch.Tensor, vp: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
#         # vm, vp: [B, C]; tab: [B, T]
#         fused = torch.cat([vm, vp, tab], dim=1)
#         logits = self.classifier(fused)
#         return logits

# -------------- Define multimodal classifier (Simple MLP example) --------------
class MultiModalClassifier(nn.Module):
    """img_feat [B, img_dim] & table [B, tab_dim] -> logits [B, num_cls]"""
    def __init__(self, img_dim: int, tab_dim: int, num_classes: int):
        super().__init__()
        self.img_dim   = img_dim
        self.tab_dim   = tab_dim
        self.num_cls   = num_classes
        self.fc = nn.Sequential(
            nn.Linear(self.img_dim + self.tab_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_cls)
        )

    def forward(self, img_feat: torch.Tensor, table_feat: torch.Tensor):
        x = torch.cat([img_feat, table_feat], dim=1)
        return self.fc(x)
