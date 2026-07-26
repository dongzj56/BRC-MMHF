"""ADNI MRI/PET loader and transforms (train: z-score+noise; val/test: z-score)."""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandZoomd,
    SpatialPadd,
)
from torch.utils.data import Dataset

from .tabular import normalize_group_for_task, resolve_task_labels

PAD_SIZE = (96, 112, 96)
IMAGE_KEYS = ("MRI", "PET")


def build_mri_pet_transforms(
    *,
    augment: bool = False,
    pad_size: Sequence[int] = PAD_SIZE,
    noise_std: float = 0.1,
    noise_prob: float = 0.5,
    keys: Sequence[str] = IMAGE_KEYS,
) -> Tuple[Compose, Compose]:
    key_list = list(keys)
    if not key_list:
        return Compose([]), Compose([])
    common = [
        LoadImaged(keys=key_list),
        EnsureChannelFirstd(keys=key_list),
        NormalizeIntensityd(keys=key_list, nonzero=True, channel_wise=True),
        SpatialPadd(keys=key_list, spatial_size=tuple(pad_size), method="end", mode="constant"),
        EnsureTyped(keys=key_list),
    ]
    train_list = list(common)
    if augment:
        spatial = [
            RandFlipd(keys=key_list, prob=0.3, spatial_axis=0),
            RandRotated(keys=key_list, prob=0.3, range_x=0.05),
            RandZoomd(keys=key_list, prob=0.3, min_zoom=0.95, max_zoom=1.0),
        ]
        train_list = common[:3] + spatial + common[3:]
    train_list = (
        train_list[:-1]
        + [RandGaussianNoised(keys=key_list, prob=noise_prob, mean=0.0, std=noise_std)]
        + train_list[-1:]
    )
    return Compose(train_list), Compose(common)


def ADNI_transform(
    augment: bool = False,
    noise_std: float = 0.1,
    noise_prob: float = 0.5,
    pad_size: Sequence[int] = PAD_SIZE,
    keys: Sequence[str] = IMAGE_KEYS,
) -> Tuple[Compose, Compose]:
    return build_mri_pet_transforms(
        augment=augment,
        pad_size=pad_size,
        noise_std=noise_std,
        noise_prob=noise_prob,
        keys=keys,
    )


def _resolve_nifti_path(directory: str, subject_id: str) -> str:
    path = os.path.join(directory, f"{subject_id}.nii")
    if os.path.isfile(path):
        return path
    gz = path + ".gz"
    return gz if os.path.isfile(gz) else path


def build_image_data_dict(
    label_df: pd.DataFrame,
    mri_dir: str,
    pet_dir: str,
    task: str,
    *,
    subject_col: str = "Subject_ID",
    label_col: str = "Group",
    check_files: bool = True,
    allow_missing: bool = False,
    dataset_name: str = "Dataset",
) -> List[dict]:
    label_dict = resolve_task_labels(task)
    if subject_col not in label_df.columns:
        raise KeyError(f"[{dataset_name}] Missing column '{subject_col}'")
    if label_col not in label_df.columns:
        raise KeyError(f"[{dataset_name}] Missing column '{label_col}'")

    groups = label_df[label_col].map(lambda g: normalize_group_for_task(g, task))
    mask = groups.isin(label_dict.keys())
    filtered = label_df.loc[mask].copy()
    filtered["_group_norm"] = groups.loc[mask]
    if filtered.empty:
        raise ValueError(f"[{dataset_name}] No samples for task '{task}'")

    dups = filtered[subject_col][filtered[subject_col].duplicated()].tolist()
    if dups:
        raise ValueError(f"[{dataset_name}] Duplicate Subject_ID: {dups[:10]}")

    samples: List[dict] = []
    missing_mri: List[str] = []
    missing_pet: List[str] = []
    for _, row in filtered.iterrows():
        sid = str(row[subject_col])
        mri_path = _resolve_nifti_path(mri_dir, sid)
        pet_path = _resolve_nifti_path(pet_dir, sid)
        if check_files:
            has_mri, has_pet = os.path.isfile(mri_path), os.path.isfile(pet_path)
            if not has_mri:
                missing_mri.append(sid)
            if not has_pet:
                missing_pet.append(sid)
            if not has_mri or not has_pet:
                continue
        samples.append(
            {
                "MRI": mri_path,
                "PET": pet_path,
                "label": int(label_dict[row["_group_norm"]]),
                "Subject": sid,
            }
        )

    if check_files and (missing_mri or missing_pet) and not allow_missing:
        msgs = []
        if missing_mri:
            msgs.append(f"missing MRI ({len(missing_mri)}): {missing_mri[:5]}")
        if missing_pet:
            msgs.append(f"missing PET ({len(missing_pet)}): {missing_pet[:5]}")
        raise FileNotFoundError(f"[{dataset_name}] {'; '.join(msgs)}")
    if not samples:
        raise ValueError(f"[{dataset_name}] Empty data_dict")
    return samples


def attach_tabular(
    data_dict: List[dict],
    table_map: Dict[str, np.ndarray],
    *,
    allow_missing: bool = False,
    tab_dim: Optional[int] = None,
) -> List[dict]:
    missing: List[str] = []
    out: List[dict] = []
    inferred_dim = tab_dim
    for sample in data_dict:
        sid = sample["Subject"]
        if sid not in table_map:
            missing.append(sid)
            if not allow_missing:
                continue
            if inferred_dim is None:
                raise ValueError("tab_dim required when allow_missing=True")
            vec = np.zeros((inferred_dim,), dtype=np.float32)
        else:
            vec = np.asarray(table_map[sid], dtype=np.float32)
            inferred_dim = int(vec.shape[0])
        item = dict(sample)
        item["tabular"] = vec
        out.append(item)
    if missing and not allow_missing:
        raise KeyError(f"Tabular missing for {len(missing)} subjects: {missing[:5]}")
    return out


class MultiModalDataset(Dataset):
    """MRI + PET (+ optional tabular) sample dicts."""

    def __init__(
        self,
        label_file: str,
        mri_dir: str,
        pet_dir: str,
        task: str = "SMCIPMCI",
        *,
        table_map: Optional[Dict[str, np.ndarray]] = None,
        check_files: bool = True,
        allow_missing: bool = False,
        augment: bool = False,
        dataset_name: str = "MultiModal",
        encoding: str = "ISO-8859-1",
    ) -> None:
        import torch

        self._torch = torch
        self.task = task
        self.augment = augment
        self.dataset_name = dataset_name
        self.label_dict = resolve_task_labels(task)
        self.label_df = pd.read_csv(label_file, encoding=encoding)
        self.data_dict = build_image_data_dict(
            self.label_df,
            mri_dir,
            pet_dir,
            task,
            check_files=check_files,
            allow_missing=allow_missing,
            dataset_name=dataset_name,
        )
        if table_map is not None:
            self.data_dict = attach_tabular(
                self.data_dict, table_map, allow_missing=allow_missing
            )
        self._print_class_counts()

    def _print_class_counts(self) -> None:
        id_to_name = {v: k for k, v in self.label_dict.items()}
        counts = Counter(s["label"] for s in self.data_dict)
        print(f"\n[{self.dataset_name}: {self.task}] n={len(self.data_dict)}")
        for cid, n in sorted(counts.items()):
            print(f"  {id_to_name.get(cid, cid)} ({cid}): {n}")
        print()

    def __len__(self) -> int:
        return len(self.data_dict)

    def __getitem__(self, idx: int) -> dict:
        sample = dict(self.data_dict[idx])
        if "tabular" in sample and not self._torch.is_tensor(sample["tabular"]):
            sample["tabular"] = self._torch.as_tensor(
                sample["tabular"], dtype=self._torch.float32
            )
        return sample


class ADNI(Dataset):
    """ADNI MRI+PET. Exposes ``data_dict`` for MMHF."""

    def __init__(
        self,
        label_file: str,
        mri_dir: str,
        pet_dir: str,
        task: str = "ADCN",
        augment: bool = False,
        *,
        check_files: bool = False,
        allow_missing: bool = False,
        table_map=None,
    ) -> None:
        self._ds = MultiModalDataset(
            label_file=label_file,
            mri_dir=mri_dir,
            pet_dir=pet_dir,
            task=task,
            augment=augment,
            check_files=check_files,
            allow_missing=allow_missing,
            table_map=table_map,
            dataset_name="ADNI",
        )
        self.task = task
        self.augment = augment
        self.label_df = self._ds.label_df
        self.label_dict = self._ds.label_dict
        self.data_dict = self._ds.data_dict
        allowed = set(resolve_task_labels(task))
        norm = self.label_df["Group"].map(lambda g: normalize_group_for_task(g, task))
        self.labels = self.label_df.loc[norm.isin(allowed)].copy()

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        sample = self.data_dict[idx]
        mri = LoadImaged(keys=["MRI"])({"MRI": sample["MRI"]})["MRI"]
        pet = LoadImaged(keys=["PET"])({"PET": sample["PET"]})["PET"]
        return mri, pet, sample["label"]
