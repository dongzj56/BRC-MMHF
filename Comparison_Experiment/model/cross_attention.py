import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    Standard Cross Attention Module.
    Query comes from x, Key and Value come from y.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        """
        Args:
            x: Query source [Batch, SeqLen_x, Dim]
            y: Key/Value source [Batch, SeqLen_y, Dim]
        Returns:
            output: [Batch, SeqLen_x, Dim]
        """
        # MultiheadAttention(query, key, value)
        attn_output, _ = self.multihead_attn(x, y, y)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x

class CrossModalFusion(nn.Module):
    """
    Cross Modal Fusion using Cross Attention.
    Inputs are feature vectors from two modalities.
    """
    def __init__(self, in_dim1, in_dim2, embed_dim=256, num_heads=4, dropout=0.1):
        super(CrossModalFusion, self).__init__()
        
        # Project inputs to common embedding dimension
        self.proj1 = nn.Linear(in_dim1, embed_dim)
        self.proj2 = nn.Linear(in_dim2, embed_dim)
        
        # Cross Attention Modules
        # Modality 1 attends to Modality 2
        self.cross_attn1 = CrossAttention(embed_dim, num_heads, dropout)
        # Modality 2 attends to Modality 1
        self.cross_attn2 = CrossAttention(embed_dim, num_heads, dropout)
        
        # Final fusion (concatenation + projection or just concatenation)
        self.fusion_norm = nn.LayerNorm(embed_dim * 2)
        
    def forward(self, x1, x2):
        """
        x1: Feature vector from Modality 1 [Batch, in_dim1]
        x2: Feature vector from Modality 2 [Batch, in_dim2]
        """
        # 1. Unsqueeze to add Sequence Length dimension (Batch, 1, Dim)
        if x1.dim() == 2:
            x1 = x1.unsqueeze(1)
        if x2.dim() == 2:
            x2 = x2.unsqueeze(1)
            
        # 2. Project to embedding dimension
        h1 = self.proj1(x1) # [Batch, 1, embed_dim]
        h2 = self.proj2(x2) # [Batch, 1, embed_dim]
        
        # 3. Cross Attention
        # h1 queries h2
        feat1_fused = self.cross_attn1(h1, h2) 
        # h2 queries h1
        feat2_fused = self.cross_attn2(h2, h1)
        
        # 4. Concatenate
        fused = torch.cat([feat1_fused, feat2_fused], dim=-1) # [Batch, 1, 2*embed_dim]
        
        # 5. Flatten
        fused = fused.squeeze(1) # [Batch, 2*embed_dim]
        
        fused = self.fusion_norm(fused)
        
        return fused

if __name__ == '__main__':
    # Test
    # Suppose DenseNet121 outputs 1024 dim
    model = CrossModalFusion(in_dim1=1024, in_dim2=1024, embed_dim=256, num_heads=4)
    x1 = torch.randn(4, 1024)
    x2 = torch.randn(4, 1024)
    out = model(x1, x2)
    print("Fused Output shape:", out.shape) # Should be (4, 512)
