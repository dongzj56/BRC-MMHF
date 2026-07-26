"""Clinical tabular preprocessing (fit on train only) + task/feature schemas."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

TASK_LABELS: Dict[str, Dict[str, int]] = {
    "ADCN": {"CN": 0, "AD": 1},
    "SMCIPMCI": {"SMCI": 0, "PMCI": 1},
    "ADCNMCI": {"CN": 0, "MCI": 1, "AD": 2},
}
MCI_SOURCE_GROUPS = ("SMCI", "PMCI", "MCI")
META_COLS = ("Subject_ID", "Group", "Group2", "PTID", "label")
LEAKAGE_EXCLUDE = [
    "VENTRICLES", "HIPPOCAMPUS", "WHOLEBRAIN", "ENTORHINAL", "FUSIFORM",
    "MIDTEMP", "ICV", "FDG", "AV45", "ABETA", "TAU", "PTAU",
    "VISCODE", "EXAMDATE", "DX_bl", "DX_BL",
]
COMMON36 = [
    "AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY", "APOE4",
    "CDRSB", "ADAS11", "ADAS13", "ADASQ4", "MMSE",
    "RAVLT_IMMEDIATE", "RAVLT_LEARNING", "RAVLT_FORGETTING", "RAVLT_PERC_FORGETTING",
    "LDELTOTAL", "DIGITSCOR", "TRABSCOR", "FAQ", "MOCA",
    "ECOGPTMEM", "ECOGPTLANG", "ECOGPTVISSPAT", "ECOGPTPLAN", "ECOGPTORGAN",
    "ECOGPTDIVATT", "ECOGPTTOTAL",
    "ECOGSPMEM", "ECOGSPLANG", "ECOGSPVISSPAT", "ECOGSPPLAN", "ECOGSPORGAN",
    "ECOGSPDIVATT", "ECOGSPTOTAL", "GDTOTAL",
]
REDUCED = [
    "AGE", "PTGENDER", "PTEDUCAT", "APOE4", "CDRSB", "ADAS13", "MMSE",
    "RAVLT_IMMEDIATE", "LDELTOTAL", "FAQ", "MOCA", "NPISCORE",
]
MISSING_TOKEN = "__MISSING__"


def resolve_task_labels(task: str) -> Dict[str, int]:
    if task not in TASK_LABELS:
        raise ValueError(f"Unsupported task '{task}'. Choose from {list(TASK_LABELS)}")
    return dict(TASK_LABELS[task])


def normalize_group_for_task(group: str, task: str) -> str:
    g = str(group)
    if task == "ADCNMCI" and g in MCI_SOURCE_GROUPS:
        return "MCI"
    return g


@dataclass
class TabularBundle:
    subject_ids: np.ndarray
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]


def resolve_label_series(
    df: pd.DataFrame, label_col: str, task: str
) -> Tuple[pd.Series, pd.Series]:
    label_map = resolve_task_labels(task)
    raw = df[label_col].astype(str).copy()
    if task == "ADCNMCI":
        raw = raw.replace({g: "MCI" for g in MCI_SOURCE_GROUPS if g != "MCI"})
    mapped = raw.map(label_map)
    mask = mapped.notna()
    if not mask.any():
        raise ValueError(f"No rows for task '{task}'")
    return mapped, mask


def resolve_feature_columns(
    columns: Sequence[str],
    feature_schema: str = "full133",
    feature_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    available = [c for c in columns if c not in META_COLS and c not in LEAKAGE_EXCLUDE]
    if feature_cols is not None:
        return list(feature_cols)
    if feature_schema == "full133":
        return list(available)
    if feature_schema == "common36":
        return list(COMMON36)
    if feature_schema == "reduced":
        return list(REDUCED)
    raise ValueError(f"Unknown feature_schema '{feature_schema}'")


def _is_categorical(series: pd.Series) -> bool:
    if series.dtype == object or str(series.dtype).startswith("category"):
        return True
    if pd.api.types.is_integer_dtype(series):
        nunique = series.nunique(dropna=True)
        return 0 < nunique <= 20
    return False


class TabularPreprocessor:
    """Impute / encode / scale. Fit on train split only."""

    def __init__(
        self,
        subject_col: str = "Subject_ID",
        label_col: str = "Group",
        task: str = "SMCIPMCI",
        feature_schema: str = "full133",
        feature_cols: Optional[Sequence[str]] = None,
        missing_strategy: str = "median",
        cat_missing: str = "token",
        scale_numeric: bool = True,
        categorical_cols: Optional[Sequence[str]] = None,
    ) -> None:
        self.subject_col = subject_col
        self.label_col = label_col
        self.task = task
        self.feature_schema = feature_schema
        self.feature_cols_arg = list(feature_cols) if feature_cols is not None else None
        self.missing_strategy = missing_strategy
        self.cat_missing = cat_missing
        self.scale_numeric = scale_numeric
        self.categorical_cols_arg = (
            list(categorical_cols) if categorical_cols is not None else None
        )
        self.feature_names_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.num_imputer_: Optional[SimpleImputer] = None
        self.cat_imputer_: Optional[SimpleImputer] = None
        self.cat_encoder_: Optional[OrdinalEncoder] = None
        self.scaler_: Optional[StandardScaler] = None
        self.is_fitted_ = False

    def _prepare_frame(self, df: pd.DataFrame):
        if self.subject_col not in df.columns:
            raise KeyError(f"Missing subject column '{self.subject_col}'")
        if self.label_col not in df.columns:
            raise KeyError(f"Missing label column '{self.label_col}'")
        y, mask = resolve_label_series(df, self.label_col, self.task)
        sub = df.loc[mask].copy()
        y = y.loc[mask].values.astype(np.int64)
        subjects = sub[self.subject_col].astype(str).values
        if not self.feature_names_:
            self.feature_names_ = resolve_feature_columns(
                sub.columns, self.feature_schema, self.feature_cols_arg
            )
        for col in self.feature_names_:
            if col not in sub.columns:
                sub[col] = np.nan
        return sub[self.feature_names_].copy(), subjects, y

    def _infer_col_types(self, frame: pd.DataFrame) -> None:
        if self.categorical_cols_arg is not None:
            self.categorical_cols_ = [
                c for c in self.feature_names_ if c in self.categorical_cols_arg
            ]
        else:
            self.categorical_cols_ = [
                c for c in self.feature_names_ if _is_categorical(frame[c])
            ]
        self.numeric_cols_ = [
            c for c in self.feature_names_ if c not in self.categorical_cols_
        ]

    def fit(self, train_df: pd.DataFrame) -> "TabularPreprocessor":
        frame, _, _ = self._prepare_frame(train_df)
        self._infer_col_types(frame)
        if self.numeric_cols_:
            num = frame[self.numeric_cols_].apply(pd.to_numeric, errors="coerce")
            self.num_imputer_ = SimpleImputer(strategy=self.missing_strategy)
            num_imp = self.num_imputer_.fit_transform(num)
            if self.scale_numeric:
                self.scaler_ = StandardScaler()
                self.scaler_.fit(num_imp)
        if self.categorical_cols_:
            cat = frame[self.categorical_cols_].astype(object)
            if self.cat_missing == "token":
                cat = cat.where(cat.notna(), other=MISSING_TOKEN)
                fill = MISSING_TOKEN
            else:
                fill = None
            self.cat_imputer_ = SimpleImputer(
                strategy="most_frequent" if fill is None else "constant",
                fill_value=fill,
            )
            cat_imp = self.cat_imputer_.fit_transform(cat)
            self.cat_encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float64,
            )
            self.cat_encoder_.fit(cat_imp)
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> TabularBundle:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before transform().")
        frame, subjects, y = self._prepare_frame(df)
        parts: List[np.ndarray] = []
        out_names: List[str] = []
        if self.numeric_cols_:
            num = frame[self.numeric_cols_].apply(pd.to_numeric, errors="coerce")
            num_imp = self.num_imputer_.transform(num)
            if self.scaler_ is not None:
                num_imp = self.scaler_.transform(num_imp)
            parts.append(num_imp.astype(np.float32))
            out_names.extend(self.numeric_cols_)
        if self.categorical_cols_:
            cat = frame[self.categorical_cols_].astype(object)
            if self.cat_missing == "token":
                cat = cat.where(cat.notna(), other=MISSING_TOKEN)
            cat_enc = self.cat_encoder_.transform(
                self.cat_imputer_.transform(cat)
            ).astype(np.float32)
            for j, cats in enumerate(self.cat_encoder_.categories_):
                unk = float(len(cats))
                col = cat_enc[:, j]
                col[col < 0] = unk
                cat_enc[:, j] = col
            parts.append(cat_enc)
            out_names.extend(self.categorical_cols_)
        if not parts:
            raise RuntimeError("No features produced")
        return TabularBundle(
            subject_ids=subjects.astype(str),
            X=np.concatenate(parts, axis=1).astype(np.float32),
            y=y.astype(np.int64),
            feature_names=out_names,
        )

    def fit_transform(self, train_df: pd.DataFrame) -> TabularBundle:
        return self.fit(train_df).transform(train_df)

    def transform_to_map(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], int]:
        bundle = self.transform(df)
        mapping = {sid: bundle.X[i] for i, sid in enumerate(bundle.subject_ids)}
        return mapping, bundle.X.shape[1]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "subject_col": self.subject_col,
                "label_col": self.label_col,
                "task": self.task,
                "feature_schema": self.feature_schema,
                "feature_cols_arg": self.feature_cols_arg,
                "missing_strategy": self.missing_strategy,
                "cat_missing": self.cat_missing,
                "scale_numeric": self.scale_numeric,
                "feature_names_": self.feature_names_,
                "numeric_cols_": self.numeric_cols_,
                "categorical_cols_": self.categorical_cols_,
                "is_fitted_": self.is_fitted_,
                "num_imputer_": self.num_imputer_,
                "cat_imputer_": self.cat_imputer_,
                "cat_encoder_": self.cat_encoder_,
                "scaler_": self.scaler_,
            },
            path,
        )
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": self.task,
                    "feature_schema": self.feature_schema,
                    "n_features": len(self.feature_names_),
                    "feature_names": self.feature_names_,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "TabularPreprocessor":
        payload: Dict[str, Any] = joblib.load(path)
        obj = cls(
            subject_col=payload["subject_col"],
            label_col=payload["label_col"],
            task=payload["task"],
            feature_schema=payload["feature_schema"],
            feature_cols=payload["feature_cols_arg"],
            missing_strategy=payload["missing_strategy"],
            cat_missing=payload["cat_missing"],
            scale_numeric=payload["scale_numeric"],
        )
        for k in (
            "feature_names_",
            "numeric_cols_",
            "categorical_cols_",
            "num_imputer_",
            "cat_imputer_",
            "cat_encoder_",
            "scaler_",
            "is_fitted_",
        ):
            setattr(obj, k, payload[k])
        return obj


def load_tabular_csv(csv_path: str, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding=encoding)


def fit_transform_fold_table(
    table_df: pd.DataFrame,
    train_subject_ids: Sequence[str],
    all_subject_ids: Sequence[str],
    *,
    task: str = "SMCIPMCI",
    feature_schema: str = "full133",
    feature_cols: Optional[Sequence[str]] = None,
    scale_numeric: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], int, TabularPreprocessor]:
    """Fit on train subjects only; transform all requested subjects."""
    train_set = set(map(str, train_subject_ids))
    train_df = table_df[table_df["Subject_ID"].astype(str).isin(train_set)].copy()
    if train_df.empty:
        raise ValueError("No overlapping train subjects between images and table")
    pre = TabularPreprocessor(
        task=task,
        feature_schema=feature_schema,
        feature_cols=feature_cols,
        scale_numeric=scale_numeric,
    )
    pre.fit(train_df)
    mapping, dim = pre.transform_to_map(table_df)
    out = {
        sid: mapping.get(sid, np.zeros((dim,), dtype=np.float32))
        for sid in map(str, all_subject_ids)
    }
    if save_path:
        pre.save(save_path)
    return out, dim, pre
