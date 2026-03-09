#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

import itertools
import logging
import os
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T")

class TabPFNEstimator(Protocol):
    def fit(self, X: Any, y: Any) -> Any: ...

    def predict(self, X: Any) -> Any: ...

def is_tabpfn(estimator: Any) -> bool:
    try:
        return any(
            [
                "TabPFN" in str(estimator.__class__),
                "TabPFN" in str(estimator.__class__.__bases__),
                any("TabPFN" in str(b) for b in estimator.__class__.__bases__),
                "tabpfn.base_model.TabPFNBaseModel" in str(estimator.__class__.mro()),
            ],
        )
    except (AttributeError, TypeError):
        return False

def get_device(device: str | None = "auto") -> str:
    import torch

    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but not available. "
            "Use device='auto' to fall back to CPU automatically.",
        )
    return device

USE_TABPFN_LOCAL = os.getenv("USE_TABPFN_LOCAL", "true").lower() == "true"

try:
    from tabpfn_client import (
        TabPFNClassifier as ClientTabPFNClassifierBase,
        TabPFNRegressor as ClientTabPFNRegressorBase,
    )

    if os.getenv("TABPFN_DEBUG", "false").lower() == "true":
        logging.info("Using TabPFN client")

    class ClientTabPFNClassifier(ClientTabPFNClassifierBase):
        def __init__(
            self,
            device: str | None = None,
            categorical_features_indices: list[int] | None = None,
            model_path: str = "default",
            n_estimators: int = 4,
            softmax_temperature: float = 0.9,
            balance_probabilities: bool = False,
            average_before_softmax: bool = False,
            ignore_pretraining_limits: bool = False,
            inference_precision: Literal["autocast", "auto"] = "auto",
            random_state: int
            | np.random.RandomState
            | np.random.Generator
            | None = None,
            inference_config: dict | None = None,
            paper_version: bool = False,
        ) -> None:
            self.device = device
            self.categorical_features_indices = categorical_features_indices
            if categorical_features_indices is not None:
                warnings.warn(
                    "categorical_features_indices is not supported in the client version of TabPFN and will be ignored",
                    UserWarning,
                    stacklevel=2,
                )
            if "/" in model_path:
                model_name = model_path.split("/")[-1].split("-")[-1].split(".")[0]
                if model_name == "classifier":
                    model_name = "default"
                self.model_path = model_name
            else:
                self.model_path = model_path

            super().__init__(
                model_path=self.model_path,
                n_estimators=n_estimators,
                softmax_temperature=softmax_temperature,
                balance_probabilities=balance_probabilities,
                average_before_softmax=average_before_softmax,
                ignore_pretraining_limits=ignore_pretraining_limits,
                inference_precision=inference_precision,
                random_state=random_state,
                inference_config=inference_config,
                paper_version=paper_version,
            )

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            params = super().get_params(deep=deep)
            params.pop("device")
            params.pop("categorical_features_indices")
            return params

    class ClientTabPFNRegressor(ClientTabPFNRegressorBase):
        def __init__(
            self,
            device: str | None = None,
            categorical_features_indices: list[int] | None = None,
            model_path: str = "default",
            n_estimators: int = 8,
            softmax_temperature: float = 0.9,
            average_before_softmax: bool = False,
            ignore_pretraining_limits: bool = False,
            inference_precision: Literal["autocast", "auto"] = "auto",
            random_state: int
            | np.random.RandomState
            | np.random.Generator
            | None = None,
            inference_config: dict | None = None,
            paper_version: bool = False,
        ) -> None:
            self.device = device
            self.categorical_features_indices = categorical_features_indices
            if categorical_features_indices is not None:
                warnings.warn(
                    "categorical_features_indices is not supported in the client version of TabPFN and will be ignored",
                    UserWarning,
                    stacklevel=2,
                )

            if "/" in model_path:
                model_name = model_path.split("/")[-1].split("-")[-1].split(".")[0]
                if model_name == "regressor":
                    model_name = "default"
                self.model_path = model_name
            else:
                self.model_path = model_path

            super().__init__(
                model_path=self.model_path,
                n_estimators=n_estimators,
                softmax_temperature=softmax_temperature,
                average_before_softmax=average_before_softmax,
                ignore_pretraining_limits=ignore_pretraining_limits,
                inference_precision=inference_precision,
                random_state=random_state,
                inference_config=inference_config,
                paper_version=paper_version,
            )

        def predict(self, X, output_type=None, **kwargs):
            if output_type != "full":
                return super().predict(X)

            try:
                import torch

                from tabpfn.model.bar_distribution import FullSupportBarDistribution

                client_output = super().predict(X, output_type="full")

                criterion = FullSupportBarDistribution(
                    borders=torch.tensor(client_output["borders"]),
                )

                result = dict(client_output)
                result["criterion"] = criterion

                return result

            except ImportError:
                raise ValueError(
                    "output_type='full' requires the TabPFN package with "
                    "FullSupportBarDistribution to be installed",
                )

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            params = super().get_params(deep=deep)
            params.pop("device")
            params.pop("categorical_features_indices")
            return params

except ImportError:
    TabPFNClassifierWrapper = None
    TabPFNRegressorWrapper = None

try:
    from tabpfn import (
        TabPFNClassifier as LocalTabPFNClassifier,
        TabPFNRegressor as LocalTabPFNRegressor,
    )
except ImportError:
    LocalTabPFNClassifier = None
    LocalTabPFNRegressor = None


def get_tabpfn_models() -> tuple[type, type]:
    if USE_TABPFN_LOCAL and LocalTabPFNClassifier is not None:
        logging.info("Using TabPFN package")

        return LocalTabPFNClassifier, LocalTabPFNRegressor
    elif TabPFNClassifierWrapper is not None:
        return TabPFNClassifierWrapper, TabPFNRegressorWrapper
    else:
        raise ImportError(
            "No TabPFN implementation could be imported. Install with one of the following:\n"
            "pip install tabpfn    # For standard TabPFN package\n"
            "pip install tabpfn-client  # For TabPFN client (API-based inference)",
        )


TabPFNClassifier, TabPFNRegressor = get_tabpfn_models()


def infer_categorical_features(
    X: np.ndarray,
    categorical_features: list[int] | None = None,
) -> list[int]:
    if categorical_features is None:
        categorical_features = []

    max_unique_values_as_categorical_feature = 10
    min_unique_values_as_numerical_feature = 10

    _categorical_features: list[int] = []

    is_pandas = hasattr(X, "dtypes")

    if is_pandas:
        import pandas as pd

        for i, col_name in enumerate(X.columns):
            col = X[col_name]
            if (
                pd.api.types.is_categorical_dtype(col)
                or pd.api.types.is_object_dtype(col)
                or pd.api.types.is_string_dtype(col)
            ):
                _categorical_features.append(i)
    else:
        for i in range(X.shape[1]):
            if X.dtype == object:
                col = X[:, i]
                for val in col:
                    if val is not None and not (
                        isinstance(val, float) and np.isnan(val)
                    ):
                        if isinstance(val, str):
                            _categorical_features.append(i)
                            break

    for i in range(X.shape[-1]):
        if i in _categorical_features:
            continue

        n_unique = X.iloc[:, i].nunique() if is_pandas else len(np.unique(X[:, i]))

        if (
            i in categorical_features
            and n_unique <= max_unique_values_as_categorical_feature
        ):
            _categorical_features.append(i)

        elif (
            i not in categorical_features
            and n_unique < min_unique_values_as_numerical_feature
            and X.shape[0] > 100
        ):
            _categorical_features.append(i)

    return _categorical_features


def softmax(logits: NDArray) -> NDArray:
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)

    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)

    sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
    probs = exp_logits / sum_exp_logits

    if logits.ndim == 1:
        return probs.reshape(-1)
    return probs


def product_dict(d: dict[str, list[T]]) -> Iterator[dict[str, T]]:
    keys = d.keys()
    values = [d[key] for key in keys]
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


TabPFNClassifier, TabPFNRegressor = get_tabpfn_models()
