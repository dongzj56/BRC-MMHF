"""BRC-MMHF datasets package."""

from .tabular import COMMON36, REDUCED, TASK_LABELS, TabularBundle, TabularPreprocessor

__all__ = [
    "ADNI",
    "ADNI_transform",
    "SCAN",
    "SCAN_transform",
    "MultiModalDataset",
    "TabularPreprocessor",
    "TabularBundle",
    "make_5fold_splits",
    "save_fold_indices",
    "load_fold_indices",
    "check_dataset_integrity",
    "TASK_LABELS",
    "COMMON36",
    "REDUCED",
]


def __getattr__(name: str):
    if name in {"ADNI", "ADNI_transform", "MultiModalDataset"}:
        from .ADNI import ADNI, ADNI_transform, MultiModalDataset

        return {"ADNI": ADNI, "ADNI_transform": ADNI_transform, "MultiModalDataset": MultiModalDataset}[name]
    if name in {"SCAN", "SCAN_transform"}:
        from .SCAN import SCAN, SCAN_transform

        return SCAN if name == "SCAN" else SCAN_transform
    if name in {
        "make_5fold_splits",
        "save_fold_indices",
        "load_fold_indices",
        "check_dataset_integrity",
    }:
        from . import checks

        return getattr(checks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
