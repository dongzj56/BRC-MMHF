"""TabPFN tabular embedding utilities.

This module is optional: main training can consume a precomputed embedding CSV
through ``cfg.tabular_emb``. Use this utility only when you need to regenerate
those embeddings from the clinical table.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _load_tabpfn(device: str):
    try:
        from tabpfn_extensions import TabPFNClassifier
        from tabpfn_extensions.embedding import TabPFNEmbedding
    except Exception as exc:
        raise ImportError(
            "TabPFN embedding generation requires tabpfn_extensions. "
            "Install the same TabPFN stack used by the original project."
        ) from exc
    return TabPFNClassifier(device=device), TabPFNEmbedding


def _prepare_table(
    csv_path: str,
    label_col: str,
    classes: Sequence[str],
    *,
    feature_cols: Optional[Sequence[str]] = None,
    start_col: Optional[int] = None,
    dropna: bool = False,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    id_col = df.columns[0]
    if feature_cols is None:
        if start_col is None:
            raise ValueError("Provide feature_cols or start_col.")
        feature_cols = [c for c in df.columns[start_col:] if c != label_col]
    feature_cols = list(feature_cols)

    df = df[df[label_col].isin(classes)].copy()
    if df.empty:
        raise ValueError(f"No rows for classes {list(classes)}.")

    if dropna:
        df = df.dropna(subset=[label_col, *feature_cols])

    label_map = {name: i for i, name in enumerate(classes)}
    subjects = df[id_col].astype(str).to_numpy()
    y = df[label_col].map(label_map).astype("int64").to_numpy()

    for col in feature_cols:
        if df[col].dtype == object or str(df[col].dtype).startswith("category"):
            df[col] = pd.Categorical(df[col]).codes.astype("float32")
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32").to_numpy()
    return df, feature_cols, subjects, x, y


def build_tabpfn_embeddings(
    csv_path: str,
    out_csv: str,
    *,
    label_col: str = "Group",
    classes: Sequence[str] = ("SMCI", "PMCI"),
    feature_cols: Optional[Sequence[str]] = None,
    start_col: Optional[int] = 4,
    device: str = "cuda:0",
    dropna: bool = False,
) -> pd.DataFrame:
    """Generate one embedding CSV with ``Subject_ID``, ``label`` and embedding columns."""
    _, _, subjects, x, y = _prepare_table(
        csv_path,
        label_col,
        classes,
        feature_cols=feature_cols,
        start_col=start_col,
        dropna=dropna,
    )
    clf, embedding_cls = _load_tabpfn(device)
    embedder = embedding_cls(tabpfn_clf=clf, n_fold=0)
    emb = embedder.get_embeddings(x, y, x, data_source="train")[0]

    out = pd.DataFrame(emb)
    out.insert(0, "label", y)
    out.insert(0, "Subject_ID", subjects)
    out.to_csv(out_csv, index=False)
    return out
