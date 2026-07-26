#!/usr/bin/env python
"""Smoke-test ADNI MRI/PET/tabular loading before training.

Usage:
  python datasets/test_dataset.py
  python datasets/test_dataset.py --config config/config2.json --fold 1 --batch_size 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="ADNI dataset smoke test")
    parser.add_argument("--config", default=os.path.join("config", "config2.json"))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    try:
        from torch.utils.data import DataLoader
        from monai.data import Dataset as MonaiDataset
        from datasets.ADNI import ADNI, ADNI_transform
        from datasets.checks import make_5fold_splits, check_dataset_integrity
        from datasets.tabular import fit_transform_fold_table, load_tabular_csv
    except Exception as exc:
        print(f"[FAIL] import error: {exc}")
        return 1

    # ---- 1) config ----
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(ROOT, cfg_path)
    if not os.path.isfile(cfg_path):
        print(f"[FAIL] config not found: {cfg_path}")
        return 1
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[OK] config: {cfg_path}")

    label_file = cfg["label_file"]
    mri_dir = cfg["mri_dir"]
    pet_dir = cfg["pet_dir"]
    table_file = cfg.get("table_dir") or cfg.get("table_file")
    task = cfg.get("task", "SMCIPMCI")
    feature_schema = cfg.get("feature_schema", "full133")
    aal_path = cfg.get("AAL_dir")

    for name, path in (
        ("label_file", label_file),
        ("mri_dir", mri_dir),
        ("pet_dir", pet_dir),
        ("table_file", table_file),
    ):
        if not path or not os.path.exists(path):
            print(f"[FAIL] {name} missing: {path}")
            return 1

    # ---- 2) integrity ----
    try:
        report = check_dataset_integrity(
            label_file=label_file,
            mri_dir=mri_dir,
            pet_dir=pet_dir,
            table_file=table_file,
            task=task,
            aal_path=aal_path,
            feature_schema=feature_schema,
            sample_nifti=0,
        )
        print(report.summary())
        if report.errors:
            print(f"[FAIL] integrity errors: {report.errors}")
            return 1
    except Exception as exc:
        print(f"[FAIL] check_dataset_integrity: {exc}")
        return 1

    # ---- 3) ADNI data_dict ----
    try:
        ds = ADNI(label_file, mri_dir, pet_dir, task=task, augment=False, check_files=False)
        full_ds = ds.data_dict
    except Exception as exc:
        print(f"[FAIL] ADNI load: {exc}")
        return 1

    n = len(full_ds)
    counts = Counter(s["label"] for s in full_ds)
    print(f"[OK] ADNI samples={n}, label counts={dict(sorted(counts.items()))}")
    if task == "SMCIPMCI":
        if n != 479 or counts.get(0) != 321 or counts.get(1) != 158:
            print(
                f"[WARN] expected SMCI=321/PMCI=158/n=479, got "
                f"SMCI={counts.get(0)} PMCI={counts.get(1)} n={n}"
            )
        else:
            print("[OK] SMCIPMCI counts match paper (321/158, n=479)")

    sample0 = full_ds[0]
    for k in ("MRI", "PET", "label", "Subject"):
        if k not in sample0:
            print(f"[FAIL] data_dict missing key '{k}'")
            return 1
    print(f"[OK] sample0 keys={list(sample0.keys())} Subject={sample0['Subject']}")

    missing_mri = [s["Subject"] for s in full_ds if not os.path.isfile(s["MRI"])]
    missing_pet = [s["Subject"] for s in full_ds if not os.path.isfile(s["PET"])]
    print(f"[OK] missing MRI={len(missing_mri)}, missing PET={len(missing_pet)}")
    if missing_mri or missing_pet:
        print(f"[FAIL] example missing MRI={missing_mri[:3]} PET={missing_pet[:3]}")
        return 1

    # ---- 4) fold split + tabular fit on train only ----
    try:
        labels = [s["label"] for s in full_ds]
        folds = make_5fold_splits(
            labels,
            n_splits=int(cfg.get("n_splits", 5)),
            seed=int(cfg.get("seed", 42)),
            test_size=float(cfg.get("split_ratio_test", 0.2)),
            val_size=float(cfg.get("split_ratio_val", 0.2)),
        )
        fold_key = str(args.fold)
        if fold_key not in folds:
            print(f"[FAIL] fold {args.fold} not in splits")
            return 1
        train_idx = folds[fold_key]["train_idx"]
        val_idx = folds[fold_key]["val_idx"]
        test_idx = folds[fold_key]["test_idx"]
        print(
            f"[OK] fold {args.fold}: train/val/test="
            f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}"
        )
    except Exception as exc:
        print(f"[FAIL] make_5fold_splits: {exc}")
        return 1

    try:
        table_df = load_tabular_csv(table_file)
        train_sids = [str(full_ds[i]["Subject"]) for i in train_idx]
        all_sids = [str(s["Subject"]) for s in full_ds]
        table_map, tab_dim, _ = fit_transform_fold_table(
            table_df,
            train_sids,
            all_sids,
            task=task,
            feature_schema=feature_schema,
            scale_numeric=True,
        )
        print(f"[OK] tabular map n={len(table_map)}, tab_dim={tab_dim}, schema={feature_schema}")
    except Exception as exc:
        print(f"[FAIL] tabular preprocess: {exc}")
        return 1

    # ---- 5) one MONAI batch ----
    try:
        train_tfm, _ = ADNI_transform(
            augment=bool(cfg.get("augment", False)),
            noise_std=float(cfg.get("noise_std", 0.1)),
            noise_prob=float(cfg.get("noise_prob", 0.5)),
        )
        train_data = []
        for i in train_idx[: max(args.batch_size * 2, 4)]:
            item = dict(full_ds[i])
            sid = str(item["Subject"])
            item["tabular"] = table_map[sid]
            train_data.append(item)
        loader = DataLoader(
            MonaiDataset(train_data, transform=train_tfm),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(loader))
    except Exception as exc:
        print(f"[FAIL] DataLoader / batch: {exc}")
        return 1

    print(f"[OK] MRI shape: {tuple(batch['MRI'].shape)}")
    print(f"[OK] PET shape: {tuple(batch['PET'].shape)}")
    print(f"[OK] label: {batch['label'].tolist()}")
    subjects = batch.get("Subject", [])
    if hasattr(subjects, "tolist"):
        subjects = subjects.tolist()
    print(f"[OK] Subject: {list(subjects)}")
    print(f"[OK] tabular shape: {tuple(batch['tabular'].shape)} (tab_dim={tab_dim})")

    print("Dataset smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
