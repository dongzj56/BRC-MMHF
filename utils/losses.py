import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_class_weights(labels, num_classes=2):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    total = float(counts.sum())
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

class WeightedBinaryCrossEntropy2C(nn.Module):
    def __init__(self, class_weights=None, reduction="mean"):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, logits, targets):
        probs_pos = F.softmax(logits, dim=1)[:, 1]
        y = targets.float()
        w_pos = self.class_weights[1] if self.class_weights is not None else 1.0
        w_neg = self.class_weights[0] if self.class_weights is not None else 1.0
        loss = -(w_pos * y * torch.log(probs_pos.clamp(min=1e-8)) + w_neg * (1 - y) * torch.log((1 - probs_pos).clamp(min=1e-8)))
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class FocalLoss2C(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        probs_pos = F.softmax(logits, dim=1)[:, 1]
        y = targets.float()
        pt = torch.where(y == 1, probs_pos, 1 - probs_pos)
        if self.alpha is not None:
            a_pos = self.alpha[1]
            a_neg = self.alpha[0]
            alpha_t = torch.where(y == 1, torch.as_tensor(a_pos, device=logits.device, dtype=pt.dtype), torch.as_tensor(a_neg, device=logits.device, dtype=pt.dtype))
        else:
            alpha_t = 1.0
        loss = -alpha_t * (1 - pt).pow(self.gamma) * torch.log(pt.clamp(min=1e-8))
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def make_loss(name, device, num_classes=2, class_weights=None, **kwargs):
    if name == "weighted_bce":
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        if class_weights is not None:
            class_weights = class_weights.to(device)
        return WeightedBinaryCrossEntropy2C(class_weights=class_weights)
    if name == "ce":
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        if class_weights is not None:
            class_weights = class_weights.to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal":
        alpha = kwargs.get("alpha", class_weights)
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
        gamma = kwargs.get("gamma", 2.0)
        return FocalLoss2C(gamma=gamma, alpha=alpha)
    return nn.CrossEntropyLoss()
