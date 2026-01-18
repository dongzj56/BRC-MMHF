# Classical Deep-Learning Baseline Methods
# This code is part of a comparative experiment that covers classical deep-learning methods for disease prediction using single-modal (MRI or PET) or multi-modal imaging.

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from monai.data import Dataset as MonaiDataset

# Add project root to sys.path
# Get the directory of the current script (Comparison_Experiment)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (BRC-MMHF)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from datasets.ADNI import ADNI, ADNI_transform
from utils.metrics import calculate_metrics
from Comparison_Experiment.model.resnet import resnet18,resnet50

# -------------------- Configuration --------------------
class Config:
    def __init__(self, d):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for k, v in d.items(): 
            setattr(self, k, v)

def load_cfg(path):
    with open(path) as f: 
        return json.load(f)

config_path = "Comparison_Experiment/config/config.json"

cfg = Config(load_cfg(config_path))
os.makedirs(cfg.checkpoint_dir, exist_ok=True)

print(f"Device: {cfg.device}")
print(f"Task: {cfg.task}")
print(f"Epochs: {cfg.num_epochs}, Batch Size: {cfg.batch_size}")

# -------------------- Model Definition --------------------
class ResNetClassifier(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=2):
        super(ResNetClassifier, self).__init__()
        
        if model_name == 'resnet18':
            self.backbone = resnet18(
                sample_input_D=96, 
                sample_input_H=112, 
                sample_input_W=96, 
                num_seg_classes=num_classes
            )
            in_features = 512
        elif model_name == 'resnet50':
            self.backbone = resnet50(
                sample_input_D=96, 
                sample_input_H=112, 
                sample_input_W=96, 
                num_seg_classes=num_classes
            )
            in_features = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove the segmentation head from the backbone to save resources
        if hasattr(self.backbone, 'conv_seg'):
            del self.backbone.conv_seg
        
        # Global Average Pooling and Classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # Manually forward through the backbone layers to skip the segmentation head
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -------------------- Training Helper Functions --------------------
def train_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        mri = batch["MRI"].to(device)
        label = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            logits = model(mri)
            loss = criterion(logits, label)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * mri.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return epoch_loss, acc

def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            mri = batch["MRI"].to(device)
            label = batch["label"].to(device)
            
            with autocast(device_type='cuda'):
                logits = model(mri)
                loss = criterion(logits, label)
            
            running_loss += loss.item() * mri.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    # metrics dict keys: 'ACC', 'PRE', 'SEN', 'SPE', 'F1', 'AUC', 'MCC', 'cm'
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    return epoch_loss, metrics

# -------------------- Main Execution Loop --------------------

# 1. Load Data
full_dataset = ADNI(cfg.label_file, cfg.mri_dir, cfg.pet_dir, cfg.task, cfg.augment)
full_ds = full_dataset.data_dict
labels = [d["label"] for d in full_ds]

skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

# 2. Cross Validation
for fold, (train_idx, test_idx) in enumerate(skf.split(full_ds, labels), start=1):
    print(f"\n{'='*20} Fold {fold} {'='*20}")
    
    # Split Train/Val/Test
    train_subset = [full_ds[i] for i in train_idx]
    test_subset = [full_ds[i] for i in test_idx]
    
    # Inner split for validation (10% of training set)
    train_idx_inner, val_idx_inner = train_test_split(
        np.arange(len(train_idx)), test_size=0.1, stratify=[labels[i] for i in train_idx], random_state=cfg.seed
    )
    
    train_data = [train_subset[i] for i in train_idx_inner]
    val_data = [train_subset[i] for i in val_idx_inner]
    
    # Create DataLoaders
    train_tfm, _ = ADNI_transform(augment=cfg.augment)
    val_tfm, _ = ADNI_transform(augment=False)
    
    train_ds = MonaiDataset(data=train_data, transform=train_tfm)
    val_ds = MonaiDataset(data=val_data, transform=val_tfm)
    test_ds = MonaiDataset(data=test_subset, transform=val_tfm)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Initialize Model
    # Choose 'resnet18' or 'resnet50'
    model_name = 'resnet50'
    model = ResNetClassifier(model_name=model_name, num_classes=cfg.nb_class).to(cfg.device)
    
    # -------------------- Model Info & GPU Status --------------------
    print(f"Model Name: {model_name}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params:,}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(cfg.device)}")
        
        # 1. Model Memory
        torch.cuda.reset_peak_memory_stats(cfg.device)
        torch.cuda.empty_cache()
        model_memory = torch.cuda.memory_allocated(cfg.device)
        print(f"Model Memory: {model_memory / 1024**2:.2f} MB")
        
        # 2. Data & Activation Memory Analysis
        try:
            print("Analyzing memory usage with a sample batch...")
            dummy_batch = next(iter(train_loader))
            mri = dummy_batch["MRI"].to(cfg.device)
            label = dummy_batch["label"].to(cfg.device)
            
            after_data_memory = torch.cuda.memory_allocated(cfg.device)
            data_memory = after_data_memory - model_memory
            print(f"Input Data Memory (Batch Size {cfg.batch_size}): {data_memory / 1024**2:.2f} MB")
            
            # Forward pass to estimate Activation memory
            with autocast(device_type='cuda'):
                _ = model(mri)
            
            peak_memory = torch.cuda.max_memory_allocated(cfg.device)
            activation_memory = peak_memory - after_data_memory
            print(f"Activation Memory (approx): {activation_memory / 1024**2:.2f} MB")
            print(f"Total Peak Memory: {peak_memory / 1024**2:.2f} MB")
            
            # Cleanup
            del mri, label, dummy_batch
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(cfg.device)
            
        except Exception as e:
            print(f"Memory analysis failed: {e}")
            
        print(f"GPU Memory Reserved:  {torch.cuda.memory_reserved(cfg.device) / 1024**2:.2f} MB")
    # -----------------------------------------------------------------
    
    # Calculate Class Weights for Weighted Cross Entropy
    train_labels = [sample['label'] for sample in train_data]
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    
    # Weight = Total / (Num_Classes * Count)
    class_weights = torch.tensor(
        [total_samples / (len(class_counts) * c) for c in class_counts], 
        dtype=torch.float
    ).to(cfg.device)
    
    print(f"Class Counts: {class_counts}")
    print(f"Class Weights: {class_weights}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler('cuda')
    
    best_val_acc = 0.0
    best_model_path = os.path.join(cfg.checkpoint_dir, f"resnet_fold{fold}_best.pth")
    
    # Training Loop
    for epoch in range(cfg.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, criterion, cfg.device)
        scheduler.step()
        val_loss, val_metrics = val_epoch(model, val_loader, criterion, cfg.device)
        
        # Using uppercase keys as returned by utils.metrics.calculate_metrics
        print(f"Epoch {epoch+1}/{cfg.num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_metrics['ACC']:.4f} AUC: {val_metrics['AUC']:.4f}")
        
        if val_metrics['ACC'] > best_val_acc:
            best_val_acc = val_metrics['ACC']
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved (Acc: {best_val_acc:.4f})")
            
    # Final Test
    print(f"Testing Fold {fold}...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_metrics = val_epoch(model, test_loader, criterion, cfg.device)
    print(f"Fold {fold} Test Result:")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

# -------------------- SMCIPMCI Task Execution --------------------

# Switch task to SMCIPMCI
cfg.task = 'SMCIPMCI'
print(f"Switched Task to: {cfg.task}")

# 1. Load Data (Reload with new task)
full_dataset = ADNI(cfg.label_file, cfg.mri_dir, cfg.pet_dir, cfg.task, cfg.augment)
full_ds = full_dataset.data_dict
labels = [d["label"] for d in full_ds]

skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

# 2. Cross Validation
for fold, (train_idx, test_idx) in enumerate(skf.split(full_ds, labels), start=1):
    print(f"\n{'='*20} Fold {fold} (SMCIPMCI) {'='*20}")
    
    # Split Train/Val/Test
    train_subset = [full_ds[i] for i in train_idx]
    test_subset = [full_ds[i] for i in test_idx]
    
    # Inner split for validation (10% of training set)
    train_idx_inner, val_idx_inner = train_test_split(
        np.arange(len(train_idx)), test_size=0.1, stratify=[labels[i] for i in train_idx], random_state=cfg.seed
    )
    
    train_data = [train_subset[i] for i in train_idx_inner]
    val_data = [train_subset[i] for i in val_idx_inner]
    
    # Create DataLoaders
    train_tfm, _ = ADNI_transform(augment=cfg.augment)
    val_tfm, _ = ADNI_transform(augment=False)
    
    train_ds = MonaiDataset(data=train_data, transform=train_tfm)
    val_ds = MonaiDataset(data=val_data, transform=val_tfm)
    test_ds = MonaiDataset(data=test_subset, transform=val_tfm)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Initialize Model
    model_name = 'resnet50'
    model = ResNetClassifier(model_name=model_name, num_classes=cfg.nb_class).to(cfg.device)
    
    # Calculate Class Weights for Weighted Cross Entropy
    # sMCI (0) count vs pMCI (1) count in training set
    train_labels = [sample['label'] for sample in train_data]
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    
    # Weight = Total / (Num_Classes * Count)
    # This balances the impact of each class
    class_weights = torch.tensor(
        [total_samples / (len(class_counts) * c) for c in class_counts], 
        dtype=torch.float
    ).to(cfg.device)
    
    print(f"Class Counts: {class_counts}")
    print(f"Class Weights: {class_weights}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
    
    # Use Weighted Cross Entropy Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler('cuda')
    
    best_val_acc = 0.0
    # Update checkpoint filename for SMCIPMCI
    best_model_path = os.path.join(cfg.checkpoint_dir, f"resnet_SMCIPMCI_fold{fold}_best.pth")
    
    # Training Loop
    for epoch in range(cfg.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, criterion, cfg.device)
        scheduler.step()
        val_loss, val_metrics = val_epoch(model, val_loader, criterion, cfg.device)
        
        print(f"Epoch {epoch+1}/{cfg.num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_metrics['ACC']:.4f} AUC: {val_metrics['AUC']:.4f}")
        
        if val_metrics['ACC'] > best_val_acc:
            best_val_acc = val_metrics['ACC']
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved (Acc: {best_val_acc:.4f})")
            
    # Final Test
    print(f"Testing Fold {fold} (SMCIPMCI)...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_metrics = val_epoch(model, test_loader, criterion, cfg.device)
    print(f"Fold {fold} Test Result:")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
