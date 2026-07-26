"""5-fold splits (~60/20/20) and dataset integrity CLI."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .tabular import (
    TASK_LABELS,
    normalize_group_for_task,
    resolve_feature_columns,
    resolve_task_labels,
)


def make_5fold_splits(
    labels: Sequence[Union[int, float]],
    *,
    n_splits: int = 5,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Dict[str, Dict[str, List[int]]]:
    labels_arr = np.asarray(labels)
    if len(labels_arr) == 0:
        raise ValueError("labels is empty")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1")
    val_of_trainval = val_size / (1.0 - test_size)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result: Dict[str, Dict[str, List[int]]] = {}
    for fold_idx, (train_val_idx, test_idx) in enumerate(
        skf.split(np.arange(len(labels_arr)), labels_arr), start=1
    ):
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_of_trainval,
            random_state=seed + fold_idx,
            stratify=labels_arr[train_val_idx],
        )
        result[str(fold_idx)] = {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
        }
    return result


def save_fold_indices(folds: Dict[str, Dict[str, List[int]]], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)
    return path


def load_fold_indices(path: str, fold: Optional[Union[int, str]] = None):
    with open(path, "r", encoding="utf-8") as f:
        all_folds = json.load(f)
    if fold is None:
        return all_folds
    key = str(fold)
    if key not in all_folds:
        raise KeyError(f"Fold '{key}' not found in {path}")
    return all_folds[key]


@dataclass
class IntegrityReport:
    task: str
    n_label_rows: int = 0
    n_task_subjects: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_subjects: List[str] = field(default_factory=list)
    missing_mri: List[str] = field(default_factory=list)
    missing_pet: List[str] = field(default_factory=list)
    missing_tabular: List[str] = field(default_factory=list)
    n_aligned: int = 0
    atlas_path: Optional[str] = None
    atlas_exists: bool = False
    atlas_shape: Optional[tuple] = None
    sample_shapes: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Task: {self.task}",
            f"Label rows: {self.n_label_rows}",
            f"Task subjects: {self.n_task_subjects}",
            f"Class counts: {self.class_counts}",
            f"Aligned: {self.n_aligned}",
            f"Missing MRI/PET/tabular: {len(self.missing_mri)}/"
            f"{len(self.missing_pet)}/{len(self.missing_tabular)}",
            f"Duplicate Subject_ID: {len(self.duplicate_subjects)}",
        ]
        if self.atlas_path:
            lines.append(
                f"AAL: exists={self.atlas_exists} shape={self.atlas_shape}"
            )
        if self.warnings:
            lines.extend(f"  warn: {w}" for w in self.warnings)
        if self.errors:
            lines.extend(f"  error: {e}" for e in self.errors)
        return "\n".join(lines)


def _nifti_path(directory: str, sid: str) -> str:
    p = os.path.join(directory, f"{sid}.nii")
    if os.path.isfile(p):
        return p
    gz = p + ".gz"
    return gz if os.path.isfile(gz) else p


def check_dataset_integrity(
    label_file: str,
    mri_dir: str,
    pet_dir: str,
    *,
    table_file: Optional[str] = None,
    task: str = "SMCIPMCI",
    aal_path: Optional[str] = None,
    feature_schema: str = "full133",
    sample_nifti: int = 0,
    encoding: str = "ISO-8859-1",
) -> IntegrityReport:
    report = IntegrityReport(task=task)
    label_dict = resolve_task_labels(task)
    label_df = pd.read_csv(label_file, encoding=encoding)
    report.n_label_rows = len(label_df)

    if "Subject_ID" not in label_df.columns or "Group" not in label_df.columns:
        report.errors.append("label CSV needs Subject_ID and Group")
        return report

    report.duplicate_subjects = (
        label_df["Subject_ID"][label_df["Subject_ID"].duplicated()].astype(str).tolist()
    )
    if report.duplicate_subjects:
        report.errors.append(f"Duplicate Subject_ID: {report.duplicate_subjects[:5]}")

    groups = label_df["Group"].map(lambda g: normalize_group_for_task(g, task))
    task_df = label_df.loc[groups.isin(label_dict.keys())].copy()
    task_df["_g"] = groups.loc[groups.isin(label_dict.keys())]
    report.n_task_subjects = len(task_df)
    report.class_counts = Counter(task_df["_g"].tolist())

    table_ids = None
    if table_file:
        if not os.path.isfile(table_file):
            report.errors.append(f"table_file not found: {table_file}")
        else:
            table_df = pd.read_csv(table_file, encoding=encoding)
            if "Subject_ID" not in table_df.columns:
                report.errors.append("table CSV missing Subject_ID")
            else:
                table_ids = set(table_df["Subject_ID"].astype(str))
                missing_feats = [
                    c
                    for c in resolve_feature_columns(
                        table_df.columns, feature_schema=feature_schema
                    )
                    if c not in table_df.columns
                ]
                if missing_feats:
                    report.warnings.append(
                        f"Missing {len(missing_feats)} schema cols (ok for SCAN)"
                    )

    for _, row in task_df.iterrows():
        sid = str(row["Subject_ID"])
        ok_mri = os.path.isfile(_nifti_path(mri_dir, sid))
        ok_pet = os.path.isfile(_nifti_path(pet_dir, sid))
        if not ok_mri:
            report.missing_mri.append(sid)
        if not ok_pet:
            report.missing_pet.append(sid)
        ok_tab = True if table_ids is None else sid in table_ids
        if table_ids is not None and not ok_tab:
            report.missing_tabular.append(sid)
        if ok_mri and ok_pet and ok_tab:
            report.n_aligned += 1

    if aal_path:
        report.atlas_path = aal_path
        report.atlas_exists = os.path.isfile(aal_path)
        if report.atlas_exists:
            try:
                import nibabel as nib

                report.atlas_shape = tuple(nib.load(aal_path).shape)
            except Exception as exc:  # noqa: BLE001
                report.warnings.append(f"AAL read failed: {exc}")
        else:
            report.warnings.append(f"AAL not found: {aal_path}")

    if sample_nifti > 0:
        try:
            import nibabel as nib
        except ImportError:
            report.warnings.append("nibabel missing; skip NIfTI sampling")
        else:
            checked = 0
            for _, row in task_df.iterrows():
                if checked >= sample_nifti:
                    break
                sid = str(row["Subject_ID"])
                mri_p, pet_p = _nifti_path(mri_dir, sid), _nifti_path(pet_dir, sid)
                if not (os.path.isfile(mri_p) and os.path.isfile(pet_p)):
                    continue
                try:
                    mri, pet = nib.load(mri_p), nib.load(pet_p)
                    report.sample_shapes.append(
                        {
                            "Subject": sid,
                            "mri_shape": tuple(mri.shape),
                            "pet_shape": tuple(pet.shape),
                            "shape_match": mri.shape == pet.shape,
                        }
                    )
                    checked += 1
                except Exception as exc:  # noqa: BLE001
                    report.warnings.append(f"NIfTI failed {sid}: {exc}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Check multimodal dataset integrity")
    p.add_argument("--label_file", required=True)
    p.add_argument("--mri_dir", required=True)
    p.add_argument("--pet_dir", required=True)
    p.add_argument("--table_file", default=None)
    p.add_argument("--task", default="SMCIPMCI", choices=list(TASK_LABELS))
    p.add_argument("--aal_path", default=None)
    p.add_argument("--feature_schema", default="full133")
    p.add_argument("--sample_nifti", type=int, default=0)
    args = p.parse_args(argv)
    report = check_dataset_integrity(
        label_file=args.label_file,
        mri_dir=args.mri_dir,
        pet_dir=args.pet_dir,
        table_file=args.table_file,
        task=args.task,
        aal_path=args.aal_path,
        feature_schema=args.feature_schema,
        sample_nifti=args.sample_nifti,
    )
    print(report.summary())
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
