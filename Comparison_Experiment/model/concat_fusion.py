import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    """
    Simple Concatenation Fusion.
    Concatenates two feature vectors along the last dimension.
    """
    def __init__(self, input_dim1=None, input_dim2=None, output_dim=None):
        super(ConcatFusion, self).__init__()
        # If we want to project after concatenation
        if input_dim1 is not None and input_dim2 is not None and output_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(input_dim1 + input_dim2, output_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.projection = None

    def forward(self, x1, x2):
        """
        x1: [Batch, dim1]
        x2: [Batch, dim2]
        """
        if x1.dim() > 2:
            x1 = torch.flatten(x1, 1)
        if x2.dim() > 2:
            x2 = torch.flatten(x2, 1)
            
        fused = torch.cat([x1, x2], dim=1)
        
        if self.projection is not None:
            fused = self.projection(fused)
            
        return fused

if __name__ == '__main__':
    model = ConcatFusion()
    x1 = torch.randn(4, 512)
    x2 = torch.randn(4, 256)
    out = model(x1, x2)
    print("Fused shape:", out.shape) # (4, 768)
