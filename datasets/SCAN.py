"""SCAN MRI/PET loader (external validation). Same API as ADNI."""

from __future__ import annotations

from typing import Sequence, Tuple

from monai.transforms import Compose, LoadImaged
from torch.utils.data import Dataset

from .ADNI import PAD_SIZE, MultiModalDataset, build_mri_pet_transforms
from .tabular import normalize_group_for_task, resolve_task_labels


class SCAN(Dataset):
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
            dataset_name="SCAN",
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


def SCAN_transform(
    augment: bool = False,
    noise_std: float = 0.1,
    noise_prob: float = 0.5,
    pad_size: Sequence[int] = PAD_SIZE,
) -> Tuple[Compose, Compose]:
    return build_mri_pet_transforms(
        augment=augment, pad_size=pad_size, noise_std=noise_std, noise_prob=noise_prob
    )
