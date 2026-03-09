import os
import numpy as np
import pandas as pd
import torch
from time import perf_counter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from tqdm import tqdm

try:
    from tabpfn import TabPFNClassifier
except Exception:
    from tabpfn_extensions import TabPFNClassifier

import shapiq

CSV_PATH    = r"model_tabular\shap\ADNI_Tabel.csv"
LABEL_COL   = "Group"
CLASSES     = ["SMCI", "PMCI"]
START_COL   = 3
RANDOM_SEED = 42
N_FOLDS     = 5
MODEL_PATH  = "auto"

EXPLAIN_MAX_EVAL = 96
SHAP_BUDGET      = 256
TOPK_PRINT       = 30
OUT_CSV          = "tabpfn_shap_importance_raw.csv"
OUT_CSV_PARTIAL  = "tabpfn_shap_importance_raw_partial.csv"

OUT_PRED_CSV     = "tabpfn_predictions.csv"
OUT_SHAP_VALUES  = "tabpfn_shap_values.csv"

SAVE_EVERY       = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {DEVICE} (CUDA available: {torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"[info] GPU: {torch.cuda.get_device_name(0)}")


def section(msg: str):
    print("\n" + "=" * 12 + f" {msg} " + "=" * 12, flush=True)


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    t0 = perf_counter()

    section("Data Loading and Cleaning")
    t = perf_counter()
    df = pd.read_csv(CSV_PATH)

    raw_cols = list(df.columns[START_COL:])

    labels_raw = df[LABEL_COL].astype(str).str.strip()

    mapping = {CLASSES[0]: 0, CLASSES[1]: 1}
    y_map = labels_raw.map(mapping)

    if y_map.isna().any():
        lower_map = {CLASSES[0].lower(): 0, CLASSES[1].lower(): 1}
        y_map2 = labels_raw.str.lower().map(lower_map)
        y_map = y_map.fillna(y_map2)

    if y_map.isna().any():
        unmapped_vals = labels_raw[y_map.isna()].value_counts()
        print("[warn] Found unmapped label values (will be dropped):")
        print(unmapped_vals.to_string())
        print("[hint] To include them, adjust CLASSES or filter beforehand.")

    keep_mask = ~y_map.isna()
    dropped = (~keep_mask).sum()
    if dropped > 0:
        print(f"[info] Samples dropped due to unmapped/missing labels: {dropped}")
    df = df.loc[keep_mask].reset_index(drop=True)
    y_all = y_map.loc[keep_mask].astype(int).values

    if np.unique(y_all).size < 2:
        raise ValueError(f"Only one class remaining: {np.unique(y_all)}. Please check LABEL_COL/CLASSES or filtering strategy.")

    X_df = df[raw_cols].copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(X_df.values).astype(np.float32)

    rng = np.ptp(X_all, axis=0)
    nonconst_mask = rng > 1e-12
    if nonconst_mask.sum() < len(nonconst_mask):
        removed = [c for c, keep in zip(raw_cols, nonconst_mask) if not keep]
        print(f"[info] Filtered {len(removed)} zero-variance features: {removed[:10]}{' ...' if len(removed)>10 else ''}")
    X_all = X_all[:, nonconst_mask]
    raw_cols = [c for c, keep in zip(raw_cols, nonconst_mask) if keep]

    if not np.isfinite(X_all).all():
        raise ValueError("NaN/Inf still exists after imputation, please check data.")

    print(f"[done] Time elapsed: {(perf_counter()-t):.2f}s · Data shape: X={X_all.shape}, y={y_all.shape[0]}")

    section("5-Fold Cross-Validation")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    metrics_rows = []
    base_out_csv, out_csv_ext = os.path.splitext(OUT_CSV)
    base_out_csv_partial, out_csv_partial_ext = os.path.splitext(OUT_CSV_PARTIAL)
    base_pred_csv, pred_csv_ext = os.path.splitext(OUT_PRED_CSV)
    base_shap_values, shap_values_ext = os.path.splitext(OUT_SHAP_VALUES)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), start=1):
        section(f"Fold {fold_idx}/{N_FOLDS}")
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        print(f"Train: {X_train.shape}, Test: {X_test.shape}, Pos rate(train)={y_train.mean():.3f}")

        t = perf_counter()
        model = TabPFNClassifier(device=DEVICE, model_path=MODEL_PATH, n_estimators=1)
        model.fit(X_train, y_train)
        print(f"[done] Training time {(perf_counter()-t):.2f}s")

        t = perf_counter()
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        sen = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1  = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = float("nan")

        print(f"[TabPFN] ACC={acc:.4f} PRE={pre:.4f} SEN={sen:.4f} SPE={spe:.4f} F1={f1:.4f} AUC={auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))
        print(f"[done] Evaluation time {(perf_counter()-t):.2f}s")

        pred_path = f"{base_pred_csv}_fold{fold_idx}{pred_csv_ext}"
        pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob
        }).to_csv(pred_path, index=False, encoding="utf-8-sig")
        print(f"[save] Predictions saved: {os.path.abspath(pred_path)}")

        metrics_rows.append({
            "fold": fold_idx,
            "acc": acc,
            "pre": pre,
            "sen": sen,
            "spe": spe,
            "f1": f1,
            "auc": auc
        })

        section("SHAP (Raw Columns · TabPFNExplainer · With Progress)")
        t = perf_counter()
        explainer = shapiq.Explainer(
            model=model,
            data=X_train,
            labels=y_train,
            index="SV",
            max_order=1,
        )

        n_eval = min(EXPLAIN_MAX_EVAL, len(X_test))
        if n_eval <= 0:
            print("[warn] Not enough test samples, skipping SHAP.")
            continue

        X_explain = X_test[:n_eval]
        abs_shap_sum = np.zeros(len(raw_cols), dtype=np.float64)
        shap_rows = []

        def save_partial(done):
            if done <= 0:
                return
            mean_abs_shap_partial = abs_shap_sum / done
            out_partial = f"{base_out_csv_partial}_fold{fold_idx}{out_csv_partial_ext}"
            (pd.DataFrame({
                "feature": raw_cols,
                "mean_abs_shap_partial": mean_abs_shap_partial
            }).sort_values("mean_abs_shap_partial", ascending=False)
             .to_csv(out_partial, index=False, encoding="utf-8-sig"))

        print(f"[info] n_eval={n_eval}, budget={SHAP_BUDGET}")
        skipped = 0
        avg_t = None
        with tqdm(total=n_eval, desc="SHAP explaining", unit="sample", miniters=1, mininterval=0.1) as pbar:
            for i in range(n_eval):
                t1 = perf_counter()
                try:
                    sv = explainer.explain(X_explain[i], budget=SHAP_BUDGET)
                    shap_vec = np.zeros(len(raw_cols), dtype=np.float64)
                    for k, v in sv.dict_values.items():
                        if isinstance(k, tuple) and len(k) == 1:
                            shap_vec[k[0]] = v
                    abs_shap_sum += np.abs(shap_vec)

                    vals = X_explain[i]
                    shap_rows.extend([
                        {
                            "fold": fold_idx,
                            "sample": i,
                            "feature_idx": j,
                            "feature": raw_cols[j],
                            "shap": float(shap_vec[j]),
                            "value": float(vals[j]),
                        }
                        for j in range(len(raw_cols))
                    ])

                except ValueError as e:
                    if "All features are constant" in str(e):
                        skipped += 1
                        print(f"[warn] Sample {i+1}/{n_eval} encountered constant subset, skipped.", flush=True)
                    else:
                        raise
                finally:
                    dt = perf_counter() - t1
                    avg_t = dt if avg_t is None else (0.9 * avg_t + 0.1 * dt)
                    remain = (n_eval - (i + 1)) * (avg_t if avg_t else dt)
                    pbar.update(1)
                    pbar.set_postfix(last_s=f"{dt:.2f}", avg_s=f"{avg_t:.2f}", eta_s=f"{remain:.0f}")
                    if SAVE_EVERY and (i + 1) % SAVE_EVERY == 0:
                        save_partial(i + 1 - skipped)

        effective = max(1, n_eval - skipped)
        mean_abs_shap = abs_shap_sum / effective
        imp_df = pd.DataFrame({"feature": raw_cols, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        print(f"\n[SHAP · RAW] Top-{TOPK_PRINT} features (mean |SHAP| over {effective} effective samples):")
        print(imp_df.head(TOPK_PRINT).to_string(index=False))

        out_imp = f"{base_out_csv}_fold{fold_idx}{out_csv_ext}"
        imp_df.to_csv(out_imp, index=False, encoding="utf-8-sig")
        print(f"[done] SHAP completed · Time: {(perf_counter()-t):.2f}s · Saved: {os.path.abspath(out_imp)}")

        if shap_rows:
            out_shap = f"{base_shap_values}_fold{fold_idx}{shap_values_ext}"
            pd.DataFrame(shap_rows).to_csv(out_shap, index=False, encoding="utf-8-sig")
            print(f"[save] SHAP details saved: {os.path.abspath(out_shap)}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = "tabpfn_cv_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"[save] 5-fold metrics saved: {os.path.abspath(metrics_path)}")

    section("All Completed")
    print(f"Total time: {(perf_counter()-t0):.2f}s")


if __name__ == "__main__":
    main()
