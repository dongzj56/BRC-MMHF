# models/fc_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCClassifier(nn.Module):
    def __init__(self,
                 in_dim: int,
                 num_classes: int = 2,
                 hidden_dims = (256, 128, 64),
                 p_drop: float = 0.2):
        super().__init__()

        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims)-1):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_drop)
            ]
        self.backbone = nn.Sequential(*layers)          # FC Layers
        self.head     = nn.Linear(dims[-1], num_classes)  # Classification

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        """
        x : [B, in_dim]
        returns logits : [B, num_classes]
        """
        x = self.backbone(x)
        return self.head(x)
