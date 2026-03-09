
import torch
import sys
import os

checkpoint_path = r"model_tabular\models\tab_encoder.ckpt"

try:
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get("config", {})
    print("Config keys:", config.keys())
    if "max_num_features" in config:
        print("max_num_features:", config["max_num_features"])
    else:
        print("max_num_features is MISSING")
        
    # Check other related keys to infer the correct value
    print("emsize:", config.get("emsize"))
    print("nhead:", config.get("nhead"))
    
except Exception as e:
    print(f"Error loading checkpoint: {e}")
