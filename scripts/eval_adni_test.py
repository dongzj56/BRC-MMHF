#!/usr/bin/env python
"""Evaluate saved ADNI fold checkpoints on that fold's held-out test set (no training)."""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from MMHF import (
    Cfg,
    ensure_fold_indices,
    get_dataloaders,
    infer_fold,
    load_cfg,
    load_fold_indices,
    resolve_device,
    resolve_table_map,
)
from datasets.ADNI import ADNI


def parse_folds(fold_arg: str, n_splits: int):
    if fold_arg.lower() == "all":
        return list(range(1, n_splits + 1))
    return [int(x) for x in fold_arg.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ADNI test-set evaluation on trained BRC-MMHF checkpoints"
    )
    parser.add_argument("--config", default="config/config2.json")
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Directory containing best_model_fold{k}.pth (default: config.checkpoint_dir)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional single .pth path. If set, only that fold is evaluated (--fold required unless filename has fold id).",
    )
    parser.add_argument("--fold", default="all", help="all | 1 | 1,2,3")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write test_results_eval.txt / result_eval.csv (default: checkpoint_dir)",
    )
    args = parser.parse_args()

    cfg = Cfg(load_cfg(args.config))
    cfg.device = resolve_device(getattr(cfg, "device", "cuda:0"))
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)

    out_dir = args.output_dir or cfg.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)

    full_ds = ADNI(cfg.label_file, cfg.mri_dir, cfg.pet_dir, cfg.task, cfg.augment).data_dict
    # Keep the same fold split as training (from checkpoint_dir)
    ensure_fold_indices(cfg, full_ds)
    indices_path = os.path.join(cfg.checkpoint_dir, "fold_indices.json")

    folds = parse_folds(args.fold, int(cfg.n_splits))
    if args.checkpoint:
        if len(folds) != 1:
            # try parse fold from filename best_model_fold1.pth
            base = os.path.basename(args.checkpoint)
            if "fold" in base:
                try:
                    folds = [int(base.split("fold")[1].split(".")[0])]
                except Exception as exc:
                    raise ValueError("When using --checkpoint, pass a single --fold") from exc
            else:
                raise ValueError("When using --checkpoint, pass a single --fold")

    results_txt = os.path.join(out_dir, "test_results_eval.txt")
    result_csv = os.path.join(out_dir, "result_eval.csv")
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write("Fold\tACC\tPRE\tSEN\tSPE\tBACC\tF1\tAUC\tMCC\n")
    with open(result_csv, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow([
            "fold", "idx_in_fold", "sample_id", "true_label", "pred_label", "prob_pmci", "correct"
        ])

    all_metrics = []
    for fold in folds:
        if args.checkpoint and len(folds) == 1:
            ckpt_path = args.checkpoint
        else:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"best_model_fold{fold}.pth")
        if not os.path.isfile(ckpt_path):
            print(f"[Skip] fold {fold}: checkpoint not found: {ckpt_path}")
            continue

        print(f"=== Eval fold {fold} | ckpt={ckpt_path} ===")
        train_idx, _, _ = load_fold_indices(indices_path, fold)
        table_map, tab_dim = resolve_table_map(cfg, full_ds, train_idx, fold)
        # Prefer tab_dim stored in checkpoint if present
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "tab_dim" in ckpt:
            tab_dim = int(ckpt["tab_dim"])

        _, _, loader_te = get_dataloaders(cfg, full_ds, fold)
        metrics, y_true, y_pred, y_prob = infer_fold(
            cfg, fold, loader_te, table_map, tab_dim, ckpt_path
        )
        all_metrics.append(metrics)

        with open(results_txt, "a", encoding="utf-8") as f:
            f.write(
                f"{fold}\t{metrics['ACC']:.4f}\t{metrics['PRE']:.4f}\t{metrics['SEN']:.4f}\t"
                f"{metrics['SPE']:.4f}\t{metrics['BACC']:.4f}\t{metrics['F1']:.4f}\t"
                f"{metrics['AUC']:.4f}\t{metrics['MCC']:.4f}\n"
            )
        with open(result_csv, "a", newline="", encoding="utf-8") as csv_f:
            writer = csv.writer(csv_f)
            test_data = loader_te.dataset.data
            for idx, sample_dict in enumerate(test_data):
                sid = sample_dict.get("Subject") or os.path.basename(
                    sample_dict.get("MRI", f"s{idx}")
                )
                writer.writerow([
                    fold, idx, sid, int(y_true[idx]), int(y_pred[idx]),
                    float(y_prob[idx]), int(y_true[idx] == y_pred[idx]),
                ])
        print(
            f"Fold {fold} | ACC={metrics['ACC']:.4f} BACC={metrics['BACC']:.4f} "
            f"AUC={metrics['AUC']:.4f} F1={metrics['F1']:.4f}"
        )

    if not all_metrics:
        print("No folds evaluated. Check --checkpoint_dir / best_model_fold*.pth")
        return 1

    print("=== Summary ===")
    for k in ["ACC", "PRE", "SEN", "SPE", "BACC", "F1", "AUC", "MCC"]:
        vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
        if not vals:
            print(f"{k}: nan")
        else:
            print(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"Wrote: {results_txt}")
    print(f"Wrote: {result_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
