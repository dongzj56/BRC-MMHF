import torch
import torch.nn as nn

class Classifier(nn.Module):
    """
    Simple MLP Classifier Head.
    Reference: models/fc_classifier.py
    """
    def __init__(self, in_dim, num_classes=2, hidden_dims=(256, 128, 64), p_drop=0.2):
        super(Classifier, self).__init__()
        
        layers = []
        current_dim = in_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=p_drop))
            current_dim = h_dim
            
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(current_dim, num_classes)
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        """
        x: [Batch, in_dim]
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

if __name__ == '__main__':
    # Test
    model = Classifier(in_dim=512, num_classes=2)
    x = torch.randn(4, 512)
    out = model(x)
    print("Logits shape:", out.shape) # (4, 2)
