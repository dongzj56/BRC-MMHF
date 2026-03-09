"""
Purpose: ADNI dataset loader and preprocessing utilities.
This module:
- Loads labels from ADNI CSV and maps groups to task-specific class ids
- Builds a data dictionary with MRI/PET NIfTI paths and subject ids
- Provides MONAI transforms to normalize intensity and pad both MRI/PET
  to a unified size of (96, 112, 96), with optional augmentation
- Offers simple dataset info printing helpers
"""
import os
import pandas as pd
from torch.utils.data import Dataset,DataLoader, Subset
from collections import Counter
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd, EnsureTyped,
    RandFlipd, RandRotated, RandZoomd, SpatialPadd, Compose
)

# Dataset class definition
class ADNI(Dataset):
    def __init__(self, label_file, mri_dir, pet_dir,task='ADCN', augment=False):
        # self.label = pd.read_csv(label_file)
        self.label = pd.read_csv(label_file, encoding='ISO-8859-1')
        self.mri_dir = mri_dir
        self.pet_dir = pet_dir
        self.task = task
        self.augment = augment

        self._process_labels()
        self._build_data_dict()
        self._print_class_counts()

    def _process_labels(self):
        """Extract labels from the CSV according to the specified task."""
        if self.task == 'ADCN':
            self.labels = self.label[(self.label['Group'] == 'AD') | (self.label['Group'] == 'CN')]
            self.label_dict = {'CN': 0, 'AD': 1}
        if self.task == 'SMCIPMCI':
            self.labels = self.label[(self.label['Group'] == 'SMCI') | (self.label['Group'] == 'PMCI')]
            self.label_dict = {'SMCI': 0, 'PMCI': 1}

    def _build_data_dict(self):
        subject_list = self.labels['Subject_ID'].tolist()
        label_list = self.labels['Group'].tolist()
        self.data_dict = [
            {
                'MRI': os.path.join(self.mri_dir, f'{subject}.nii'),
                'PET': os.path.join(self.pet_dir, f'{subject}.nii'),
                'label': self.label_dict[group],
                'Subject': subject
            } for subject, group in zip(subject_list, label_list)
        ]

    def _print_class_counts(self):
        """Print sample counts per label in the current data_dict."""
        inv = {v: k for k, v in self.label_dict.items()}
        cnt = Counter(sample['label'] for sample in self.data_dict)
        print(f"\n[ADNI Dataset: {self.task}] Sample distribution:")
        for lbl_value, num in cnt.items():
            print(f"  {inv[lbl_value]} ({lbl_value}): {num}")
        print()

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        """Return MRI, PET and label tensors for a given index."""
        sample = self.data_dict[idx]
        label = sample['label']

        # Load MRI/PET images
        mri_img = LoadImaged(keys=['MRI'])({'MRI': sample['MRI']})['MRI']
        pet_img = LoadImaged(keys=['PET'])({'PET': sample['PET']})['PET']
        return mri_img, pet_img, label

    def print_dataset_info(self, start=0, end=None):
        print(f"\nDataset Structure:\n{'=' * 40}")
        print(f"Total Samples: {len(self)}")
        print(f"Task: {self.task}")

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        end = end or len(self)
        df = pd.DataFrame(
            [
                [s['MRI'], s['label'], s['Subject']]
                for s in self.data_dict[start:end]
            ],
            columns=["MRI", "Label", "Subject"]
        )
        print(df)
        print(f"{'=' * 40}\n")

# Preprocessing: MRI & PET unified padding, normalization, optional augmentation
def ADNI_transform(augment=False):
    keys = ['MRI','PET']  # unified keys for both modalities
    pad_size = (96, 112, 96)  # target size

    base_transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityd(keys=keys),
        SpatialPadd(keys=keys, spatial_size=pad_size, method="end", mode="constant"),
        EnsureTyped(keys=keys),
    ]

    if augment:
        base_transforms.insert(3, RandFlipd(keys=keys, prob=0.3, spatial_axis=0))
        base_transforms.insert(4, RandRotated(keys=keys, prob=0.3, range_x=0.05))
        base_transforms.insert(5, RandZoomd(keys=keys, prob=0.3, min_zoom=0.95, max_zoom=1))

    train_transform = Compose(base_transforms)
    test_transform  = Compose(base_transforms[:5])  # no augmentation; normalization + padding only

    return train_transform, test_transform


def main():
    # ------------- Basic paths and task -------------
    label_filename  = rf'adni_dataset\ADNI_902.csv'
    mri_dir         = rf'adni_dataset\MRI'
    pet_dir         = rf'adni_dataset\PET'
    task            = 'SMCIPMCI'        # two-class subset (AD vs CN / SMCI vs PMCI)

    # ------------- 1) Build the full dataset once -------------
    full_dataset = ADNI(
        label_file=label_filename,
        mri_dir=mri_dir,
        pet_dir=pet_dir,
        task=task
    )   # prints sample distribution automatically

    # ------------- 2) Stratified index split -------------
    indices = list(range(len(full_dataset)))
    labels  = [full_dataset.data_dict[i]['label'] for i in indices]

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels           # keep label ratio consistent
    )

    # ------------- 3) Build subsets -------------
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset  = Subset(full_dataset, test_idx)

    # ------------- 4) Quick data check -------------
    sample_mri, sample_pet, sample_label = train_dataset[0]
    print(f"Sample MRI shape: {sample_mri.shape}, Label: {sample_label}")

    def preview(ds, name, k=5):
        print(f"\n{name} preview (first {k} samples):")
        for i in range(k):
            subj = full_dataset.data_dict[ds.indices[i]]['Subject']
            lbl  = full_dataset.data_dict[ds.indices[i]]['label']
            print(f"  idx={ds.indices[i]:>4}  Subject={subj}  Label={lbl}")
    preview(train_dataset, "Train", 20)
    preview(test_dataset,  "Test",  5)

    # ------------- 5) Transform pipeline -------------
    train_transform, _ = ADNI_transform(augment=False)
    print("\nTransforms pipeline:")
    for i, t in enumerate(train_transform.transforms):
        print(f"  {i+1:>2}. {t.__class__.__name__}")


if __name__ == '__main__':
    main()
    # run_5fold_cv()
    
