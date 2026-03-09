"""
Encodes tabular clinical data using a PFN-style encoder.
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding
from sklearn.model_selection import train_test_split
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"▶ Using device: {DEVICE}")

def tabular_encoder_classifier(
    csv_path: str,
    label_col: str,
    classes: List[str],
    feature_cols: Optional[List[str]] = None,
    start_col: Optional[int] = None,
    n_fold: int = 0,
    dropna: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    train_out: str = "train_embeddings.csv",
    test_out: str  = "test_embeddings.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates train/test PFN-style embeddings."""
    df = pd.read_csv(csv_path)

    if feature_cols is None:
        if start_col is None:
            raise ValueError("Must provide feature_cols or start_col")
        if len(df.columns) <= start_col:
            raise ValueError(f"start_col={start_col} is out of range")
        feature_cols = [c for c in df.columns[start_col:] if c != label_col]

    df = df[df[label_col].isin(classes)].copy()
    if df.empty:
        raise ValueError(f"No samples found for classes {classes}")

    mapping = {cls: idx for idx, cls in enumerate(classes)}
    df[label_col] = df[label_col].map(mapping).astype("int64")

    cat_cols = [c for c in feature_cols
                if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes.astype("int16")

    if dropna:
        df = df.dropna(subset=[label_col] + feature_cols)
    if df.empty:
        raise ValueError("Dataset is empty after cleaning")

    X = df[feature_cols].astype("float32").values
    y = df[label_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf      = TabPFNClassifier(device=DEVICE)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)

    train_emb = embedder.get_embeddings(X_tr, y_tr, X_te, data_source="train")[0]
    test_emb  = embedder.get_embeddings(X_tr, y_tr, X_te, data_source="test")[0]

    train_df = pd.DataFrame(train_emb); train_df.insert(0, "label", y_tr)
    test_df  = pd.DataFrame(test_emb);  test_df.insert(0, "label", y_te)

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out,  index=False)
    print(f"Saved to: {train_out}  /  {test_out}")

    return train_df, test_df

def tabular_encoder(
    csv_path: str,
    label_col: str,
    classes: List[str],
    feature_cols: Optional[List[str]] = None,
    start_col: Optional[int] = None,
    n_fold: int = 0,
    dropna: bool = False,
    out_csv: str = "tabular_embeddings.csv",
) -> pd.DataFrame:
    """Generates PFN-style embeddings for the full dataset."""
    df = pd.read_csv(csv_path)

    id_col_name = df.columns[0]
    ids = df[id_col_name].astype(str)

    if feature_cols is None:
        if start_col is None:
            raise ValueError("Must provide feature_cols or start_col")
        feature_cols = [c for c in df.columns[start_col:] if c != label_col]

    df = df[df[label_col].isin(classes)].copy()
    ids = ids[df.index]

    mapping = {cls: idx for idx, cls in enumerate(classes)}
    df[label_col] = df[label_col].map(mapping).astype("int64")

    cat_cols = [c for c in feature_cols
                if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes.astype("int16")

    if dropna:
        mask = df[feature_cols + [label_col]].notna().all(axis=1)
        df = df[mask]
        ids = ids[mask]

    X = df[feature_cols].astype("float32").values
    y = df[label_col].values

    clf = TabPFNClassifier(device=DEVICE)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)
    emb = embedder.get_embeddings(X, y, X, data_source="train")[0]

    df_emb = pd.DataFrame(emb)
    df_emb.insert(0, 'label', y)
    df_emb.insert(0, id_col_name, ids.values)
    df_emb.to_csv(out_csv, index=False)
    print(f"→ Saved to {out_csv}")

    return df_emb

def _prepare_tabular_data(
    csv_path: str,
    label_col: str,
    classes: List[str],
    feature_cols: Optional[List[str]] = None,
    start_col: Optional[int] = None,
    dropna: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Helper to load and clean data."""
    df = pd.read_csv(csv_path)

    if feature_cols is None:
        if start_col is None:
            raise ValueError("Must provide feature_cols or start_col")
        feature_cols = [c for c in df.columns[start_col:] if c != label_col]

    df = df[df[label_col].isin(classes)].copy()
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    df[label_col] = df[label_col].map(mapping).astype("int64")

    cat_cols = [
        c for c in feature_cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    for c in cat_cols:
        df[c] = pd.Categorical(df[c]).codes.astype("int16")

    if dropna:
        df = df.dropna(subset=[label_col] + feature_cols)
    if df.empty:
        raise ValueError("Dataset is empty after cleaning")

    return df, feature_cols

def tabular_encoder_train(
    csv_path: str,
    label_col: str,
    classes: List[str],
    feature_cols: Optional[List[str]] = None,
    start_col: Optional[int] = None,
    n_fold: int = 0,
    dropna: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    train_out: str = "train_embeddings.csv",
) -> pd.DataFrame:
    """Generates PFN-style embeddings for the training set."""
    df, feature_cols = _prepare_tabular_data(
        csv_path, label_col, classes, feature_cols, start_col, dropna
    )

    X = df[feature_cols].astype("float32").values
    y = df[label_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size,
        stratify=y, random_state=random_state
    )

    clf      = TabPFNClassifier(device=DEVICE)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)
    train_emb = embedder.get_embeddings(
        X_tr, y_tr, X_te, data_source="train"
    )[0]

    train_df = pd.DataFrame(train_emb)
    train_df.insert(0, "label", y_tr)
    train_df.to_csv(train_out, index=False)
    print(f"Saved training embeddings → {train_out}")

    return train_df

def tabular_encoder_test(
    csv_path: str,
    label_col: str,
    classes: List[str],
    feature_cols: Optional[List[str]] = None,
    start_col: Optional[int] = None,
    n_fold: int = 0,
    dropna: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    test_out: str  = "test_embeddings.csv",
) -> pd.DataFrame:
    """Generates PFN-style embeddings for the test set."""
    df, feature_cols = _prepare_tabular_data(
        csv_path, label_col, classes, feature_cols, start_col, dropna
    )

    X = df[feature_cols].astype("float32").values
    y = df[label_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size,
        stratify=y, random_state=random_state
    )

    clf      = TabPFNClassifier(device=DEVICE)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)
    test_emb = embedder.get_embeddings(
        X_tr, y_tr, X_te, data_source="test"
    )[0]

    test_df = pd.DataFrame(test_emb)
    test_df.insert(0, "label", y_te)
    test_df.to_csv(test_out, index=False)
    print(f"Saved test embeddings → {test_out}")

    return test_df


def tabular_encoder_fold(
    fold_dir: str,
    label_col: str,
    classes: List[str],
    feature_cols: Optional[List[str]] = None,
    start_col: Optional[int] = None,
    device: str = "cpu",
    n_fold: int = 0,
    dropna: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Encodes fold data (train/val/test) using PFN-style encoder."""
    paths = {split: os.path.join(fold_dir, f"{split}.csv")
             for split in ("train", "val", "test")}
    dfs   = {split: pd.read_csv(p) for split, p in paths.items()}

    subjects = {
        split: dfs[split]["Subject_ID"].values
        for split in ("train", "val", "test")
    }

    if feature_cols is None:
        if start_col is None:
            raise ValueError("Must provide feature_cols or start_col")
        cols = dfs["train"].columns.tolist()
        if start_col >= len(cols):
            raise ValueError(f"start_col={start_col} exceeds train.csv columns")
        feature_cols = [c for c in cols[start_col:] if c != label_col]

    for split, df in dfs.items():
        dfs[split] = df[df[label_col].isin(classes)].copy()
        mapping = {cls: idx for idx, cls in enumerate(classes)}
        dfs[split][label_col] = dfs[split][label_col].map(mapping).astype("int64")
        cat_cols = [
            c for c in feature_cols
            if dfs[split][c].dtype == "object" or str(dfs[split][c].dtype).startswith("category")
        ]
        for c in cat_cols:
            dfs[split][c] = pd.Categorical(dfs[split][c]).codes.astype("int16")
        if dropna:
            dfs[split] = dfs[split].dropna(subset=[label_col] + feature_cols)
        if dfs[split].empty:
            raise ValueError(f"{split}.csv is empty after cleaning")

    X_tr = dfs["train"][feature_cols].astype("float32").values
    y_tr = dfs["train"][label_col].values

    X_val = dfs["val"][feature_cols].astype("float32").values
    y_val = dfs["val"][label_col].values

    X_te = dfs["test"][feature_cols].astype("float32").values
    y_te = dfs["test"][label_col].values

    clf      = TabPFNClassifier(device=device)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)

    train_emb = embedder.get_embeddings(X_tr, y_tr, X_tr, data_source="train")[0]
    val_emb   = embedder.get_embeddings(X_tr, y_tr, X_val, data_source="test")[0]
    test_emb  = embedder.get_embeddings(X_tr, y_tr, X_te,  data_source="test")[0]

    def make_df(emb, labels, out_name):
        split = out_name.split("_", 1)[0]
        df_emb = pd.DataFrame(emb)
        df_emb.insert(0, "label", labels)
        df_emb.insert(0, "Subject_ID", subjects[split])
        path = os.path.join(fold_dir, out_name)
        df_emb.to_csv(path, index=False)
        print(f"✓ Saved {out_name} ({df_emb.shape})")
        return df_emb

    train_df = make_df(train_emb, y_tr, "train_emb.csv")
    val_df   = make_df(val_emb,   y_val, "val_emb.csv")
    test_df  = make_df(test_emb,  y_te,  "test_emb.csv")

    return train_df, val_df, test_df


if __name__ == "__main__":
    # Example parameters
    csv_path   = rf"adni_dataset\ADNI_Tabel.csv"
    label_col  = "Group"
    classes    = ["SMCI", "PMCI"]
    start_col  = 4
    n_fold     = 0
    dropna     = False
    out_csv    = "tabular_embeddings.csv"

    # 1) Run encoder
    df_emb = tabular_encoder(
        csv_path    = csv_path,
        label_col   = label_col,
        classes     = classes,
        feature_cols= None,
        start_col   = start_col,
        n_fold       = n_fold,
        dropna       = dropna,
        out_csv      = out_csv,
    )

    # 2) Preview
    print("\n--- Embedding Preview ---")
    print(df_emb.head())
    print(f"\nEmbedding shape: {df_emb.shape}")
