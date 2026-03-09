"""
Purpose: Minimal CSV-to-(X, y) loader for clinical/tabular data.
- Selects feature columns by names or by start_col
- Filters rows to the specified label classes and maps them to integers
- Encodes categorical features to integer codes
- Optionally drops rows with NaNs in label or features
- Returns numpy arrays (float32 features, int64 labels)
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

def load_tabular(csv_path: str,
                            label_col: str,
                            classes: List[str],
                            feature_cols: Optional[List[str]] = None,
                            start_col: Optional[int] = None,
                            dropna: bool = True) -> Tuple[np.ndarray, np.ndarray]:

    df = pd.read_csv(csv_path)

    if feature_cols is None:
        if start_col is None:
            raise ValueError("Either feature_cols or start_col must be provided.")
        all_cols = list(df.columns)
        if len(all_cols) <= start_col:
            raise ValueError(f"start_col={start_col} is out of column range.")
        feature_cols = [c for c in all_cols[start_col:] if c != label_col]

    missing = [c for c in feature_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[df[label_col].isin(classes)].copy()
    if df.empty:
        raise ValueError(f"No samples belonging to {classes} in '{label_col}'")

    mapping = {cls: idx for idx, cls in enumerate(classes)}
    df[label_col] = df[label_col].map(mapping).astype("int64")

    cat_cols = [
        c for c in feature_cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes.astype("int16")

    if dropna:
        df = df.dropna(subset=[label_col] + feature_cols)

    X = df[feature_cols].astype("float32").values
    y = df[label_col].values

    return X, y

