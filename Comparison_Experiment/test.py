import sys
import os
import torch

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Comparison_Experiment.model.densenet3d import densenet121_3d
from Comparison_Experiment.model.cross_attention import CrossModalFusion
from Comparison_Experiment.model.classifier import Classifier

# 1. 定义模型
mri_encoder = densenet121_3d(in_channels=1)
pet_encoder = densenet121_3d(in_channels=1)
fusion_layer = CrossModalFusion(in_dim1=1024, in_dim2=1024, embed_dim=256)
classifier = Classifier(in_dim=512, num_classes=2) # 512 = 256*2 (双向融合拼接)

# 创建假数据用于测试
# Batch=2, Channel=1, Depth=64, Height=64, Width=64
mri_img = torch.randn(2, 1, 64, 64, 64)
pet_img = torch.randn(2, 1, 64, 64, 64)

# 2. 前向传播
# mri_img: (B, 1, D, H, W)
# pet_img: (B, 1, D, H, W)
feat_mri = mri_encoder(mri_img)
feat_pet = pet_encoder(pet_img)
fused_feat = fusion_layer(feat_mri, feat_pet)
logits = classifier(fused_feat)

print("MRI Features shape:", feat_mri.shape)
print("PET Features shape:", feat_pet.shape)
print("Fused Features shape:", fused_feat.shape)
print("Logits shape:", logits.shape)
