import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)


def calculate_metrics(y_true, y_pred, y_score):
    if len(y_true) == 0:
        raise ValueError("No samples to evaluate. Check test_loader / data split.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sen = recall_score(y_true, y_pred, zero_division=0)
    spe = tn / (tn + fp + 1e-8)

    y_true_arr = np.asarray(y_true)
    if len(np.unique(y_true_arr)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, y_score)

    return {
        "ACC": accuracy_score(y_true, y_pred),
        "PRE": precision_score(y_true, y_pred, zero_division=0),
        "SEN": sen,
        "SPE": spe,
        "BACC": 0.5 * (sen + spe),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": auc,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "cm": cm,
    }
