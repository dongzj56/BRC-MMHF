#!/usr/bin/env python
"""Basic external validation: load a saved model and evaluate a dataset once."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from monai.data import Dataset as MonaiDataset
from torch.amp import autocast
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from MMHF import Cfg, get_table_tensor, load_cfg, load_table_embeddings
from datasets.ADNI import ADNI_transform
from datasets.SCAN import SCAN
from models.brc_mmhf import build_brc_mmhf_from_cfg
from utils.metrics import calculate_metrics


def _image_keys(cfg):
    keys = []
    if bool(getattr(cfg, "use_mri", True)):
        keys.append("MRI")
    if bool(getattr(cfg, "use_pet", True)):
        keys.append("PET")
    return keys


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="External validation for BRC-MMHF")
    parser.add_argument("--config", default="config/config2.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--mri_dir", required=True)
    parser.add_argument("--pet_dir", required=True)
    parser.add_argument("--tabular_emb", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = Cfg(load_cfg(args.config))
    cfg.label_file = args.label_file
    cfg.mri_dir = args.mri_dir
    cfg.pet_dir = args.pet_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    table_map, tab_dim = {}, 1
    if bool(getattr(cfg, "use_table", True)):
        emb_path = args.tabular_emb or getattr(cfg, "tabular_emb", None)
        if not emb_path or not os.path.isfile(emb_path):
            raise FileNotFoundError(
                "External validation with table enabled requires --tabular_emb "
                "matching the trained TabPFN embedding dimension."
            )
        table_map, tab_dim = load_table_embeddings(emb_path)

    ckpt = torch.load(args.checkpoint, map_location=cfg.device)
    if isinstance(ckpt, dict) and "tab_dim" in ckpt:
        tab_dim = int(ckpt["tab_dim"])
    model = build_brc_mmhf_from_cfg(cfg, tab_dim=tab_dim).to(cfg.device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    dataset = SCAN(
        cfg.label_file,
        cfg.mri_dir,
        cfg.pet_dir,
        task=cfg.task,
        augment=False,
        check_files=True,
    )
    _, tfm = ADNI_transform(keys=_image_keys(cfg))
    loader = DataLoader(
        MonaiDataset(dataset.data_dict, transform=tfm),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    y_true, y_pred, y_prob = [], [], []
    for batch in loader:
        mri = batch["MRI"].to(cfg.device) if bool(getattr(cfg, "use_mri", True)) else None
        pet = batch["PET"].to(cfg.device) if bool(getattr(cfg, "use_pet", True)) else None
        label = batch["label"].to(cfg.device).long()
        tabular = None
        if bool(getattr(cfg, "use_table", True)):
            tabular = get_table_tensor(batch.get("Subject", []), table_map, tab_dim, cfg.device, batch=batch)
        with autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=bool(cfg.fp16 and torch.cuda.is_available()),
        ):
            logits = model(mri=mri, pet=pet, tabular=tabular)["logits"]
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = (prob > 0.5).int()
        y_true.extend(label.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        y_prob.extend(prob.cpu().numpy().tolist())

    metrics = calculate_metrics(y_true, y_pred, y_prob)
    for key in ("ACC", "PRE", "SEN", "SPE", "BACC", "F1", "AUC", "MCC"):
        val = metrics[key]
        print(f"{key}: {val:.4f}" if not np.isnan(val) else f"{key}: nan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
