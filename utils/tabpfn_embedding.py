from __future__ import annotations

import numpy as np

from utils import TabPFNClassifier, TabPFNRegressor

class TabPFNEmbedding:

    def __init__(
        self,
        tabpfn_clf: TabPFNClassifier | None = None,
        tabpfn_reg: TabPFNRegressor | None = None,
        n_fold: int = 0,
    ) -> None:
        self.tabpfn_clf = tabpfn_clf
        self.tabpfn_reg = tabpfn_reg
        self.model = self.tabpfn_clf if self.tabpfn_clf is not None else self.tabpfn_reg
        self.n_fold = n_fold

        if self.model is not None:
            if "tabpfn_client" in str(self.model.__class__.__module__):
                raise ImportError(
                    "TabPFNEmbedding requires the full TabPFN implementation (pip install tabpfn). "
                    "The TabPFN client (pip install tabpfn-client) does not support embedding extraction.",
                )

            if not hasattr(self.model, "get_embeddings"):
                raise AttributeError(
                    f"The provided model of type {type(self.model)} does not have a get_embeddings method. "
                    "Make sure you're using the full TabPFN implementation (pip install tabpfn).",
                )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if self.model is None:
            raise ValueError("No model has been set.")
        self.model.fit(X_train, y_train)

    def get_embeddings(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X: np.ndarray,
        data_source: str,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model has been set.")

        if self.n_fold == 0:
            self.model.fit(X_train, y_train)
            return self.model.get_embeddings(X, data_source=data_source)
        elif self.n_fold >= 2:
            if data_source == "test":
                self.model.fit(X_train, y_train)
                return self.model.get_embeddings(X, data_source=data_source)
            else:
                from sklearn.model_selection import KFold

                kf = KFold(n_splits=self.n_fold, shuffle=False)
                embeddings = []
                for train_index, val_index in kf.split(X_train):
                    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                    y_train_fold, _y_val_fold = y_train[train_index], y_train[val_index]
                    self.model.fit(X_train_fold, y_train_fold)
                    embeddings.append(
                        self.model.get_embeddings(X_val_fold, data_source="test"),
                    )
                return np.concatenate(embeddings, axis=1)
        else:
            raise ValueError("n_fold must be greater than 1.")
