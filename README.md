# BRC-MMHF

Brain Region-Centered Multimodal Hypergraph Fusion for MCI Conversion Prediction.

This repository contains a compact PyTorch implementation for the paper workflow:
ADNI MRI, FDG-PET, and baseline clinical tabular variables for SMCI/PMCI conversion prediction.

## Requirements

Use Python 3.10. The main dependencies are listed in [env/requirements.txt](env/requirements.txt).
Conda users can start from [env/environment.yml](env/environment.yml).

## Data

The default config expects processed data under `adni_dataset/`:

- `ADNI_902.csv`
- `ADNI_Tabel.csv`
- `MRI/*.nii` or `MRI/*.nii.gz`
- `PET/*.nii` or `PET/*.nii.gz`
- `AAL_space-MNI152NLin6_res-2x2x2.nii/...`

The default task is `SMCIPMCI`, with SMCI as class `0` and PMCI as class `1`.

## Checks

Check ADNI labels, modality alignment, AAL atlas, tabular preprocessing, fold split, and one MONAI batch:

```bash
python datasets/test_dataset.py --config config/config.json --fold 1 --batch_size 4
```

Run a CPU-friendly model forward smoke test:

```bash
python scripts/test_model_forward.py --config config/config.json --device cpu
```

Generate TabPFN tabular embeddings when you want to use the original table branch:

```bash
python scripts/build_tabpfn_embeddings.py --table_csv adni_dataset/ADNI_Tabel.csv --out_csv adni_dataset/tabular_embeddings.csv
```

## Training

Run the main model:

```bash
python MMHF.py
```

Run module ablations:

```bash
python experiments/run_ablation.py --experiment module_ablation --variant no_[hattn/hgc/ce/sl]
```

Run modality ablations:

```bash
python experiments/run_ablation.py --experiment modality_ablation --variant [mri/pet/table/mri_pet/mri_table/pet_table/mri_pet_table/table36]
```

## External Validation

Load a saved checkpoint and evaluate an external dataset with the same modality settings:

```bash
python scripts/external_validate.py --config config/config.json --checkpoint checkpoints_mmad_mci/best_model_fold1.pth --label_file scan_dataset/labels.csv --mri_dir scan_dataset/MRI --pet_dir scan_dataset/PET --tabular_emb scan_dataset/tabular_embeddings.csv
```
