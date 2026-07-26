import os, json, time, csv, numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from monai.data import Dataset as MonaiDataset
from tqdm import tqdm
from utils.metrics import calculate_metrics
from datasets.ADNI import ADNI, ADNI_transform
from datasets.checks import make_5fold_splits, save_fold_indices
from datasets.tabular import fit_transform_fold_table, load_tabular_csv
from models.brc_mmhf import BRCMMHF, build_brc_mmhf_from_cfg
from utils.losses import compute_class_weights, make_loss


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Cfg:
    DEFAULTS = {
        "task": "SMCIPMCI",
        "augment": False,
        "noise_std": 0.1,
        "noise_prob": 0.5,
        "split_ratio_test": 0.2,
        "split_ratio_val": 0.2,
        "seed": 42,
        "device": "cuda:0",
        "num_epochs": 100,
        "batch_size": 1,
        "lr": 1e-6,
        "optimizer": "adam",
        "weight_decay": 1e-4,
        "fp16": True,
        "checkpoint_dir": "checkpoints_mmad_mci",
        "nb_class": 2,
        "n_splits": 5,
        "feature_schema": "full133",
        "show_batch_progress": True,
        "use_mri": True,
        "use_pet": True,
        "use_table": True,
        "use_shared_layer": True,
        "use_channel_exchange": True,
        "use_hypergraph_conv": True,
        "use_hypergraph_attention": True,
        "use_seg_task": False,
        "seg_task": False,
        "seg_alpha": 0.05,
        "roi_start": 1,
        "roi_end": 90,
        "hg_ks": [6, 18],
        "exchange_ratio": 0.2,
        "level_channels": [64, 128, 256],
        "bottleneck_channel": 512,
        "image_feature_dim": 128,
        "tab_feature_dim": 64,
        "dropout_rate": 0.5,
    }

    def __init__(self, d):
        values = dict(self.DEFAULTS)
        values.update(d)
        for k, v in values.items():
            setattr(self, k, v)
        self.device = resolve_device(getattr(self, "device", "cuda:0"))


def resolve_device(device_spec="cuda:0"):
    """Prefer cuda:0 for training; fall back to CPU only if CUDA is unavailable."""
    spec = str(device_spec).strip() if device_spec is not None else "cuda:0"
    if not torch.cuda.is_available():
        print("[Device] CUDA unavailable -> using cpu")
        return torch.device("cpu")
    if not spec.startswith("cuda"):
        # Force training onto GPU 0 unless user explicitly needs CPU for debugging
        print(f"[Device] overriding '{spec}' -> cuda:0")
        spec = "cuda:0"
    # Normalize bare "cuda" to cuda:0
    if spec == "cuda":
        spec = "cuda:0"
    idx = int(spec.split(":")[1]) if ":" in spec else 0
    if idx < 0 or idx >= torch.cuda.device_count():
        print(f"[Device] invalid {spec}, falling back to cuda:0")
        idx = 0
        spec = "cuda:0"
    torch.cuda.set_device(idx)
    print(f"[Device] using {spec} ({torch.cuda.get_device_name(idx)})")
    return torch.device(spec)


def _fold_meta(cfg, n_samples):
    return {
        "seed": int(getattr(cfg, "seed", 42)),
        "n_splits": int(getattr(cfg, "n_splits", 5)),
        "test_size": float(getattr(cfg, "split_ratio_test", 0.2)),
        "val_size": float(getattr(cfg, "split_ratio_val", 0.2)),
        "n_samples": int(n_samples),
        "task": str(getattr(cfg, "task", "")),
    }


def build_fold_indices(cfg, data_dict):
    labels = [d["label"] for d in data_dict]
    result = make_5fold_splits(
        labels,
        n_splits=int(getattr(cfg, "n_splits", 5)),
        seed=int(getattr(cfg, "seed", 42)),
        test_size=float(getattr(cfg, "split_ratio_test", 0.2)),
        val_size=float(getattr(cfg, "split_ratio_val", 0.2)),
    )
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    json_path = os.path.join(cfg.checkpoint_dir, "fold_indices.json")
    save_fold_indices(result, json_path)
    meta_path = os.path.join(cfg.checkpoint_dir, "fold_indices.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_fold_meta(cfg, len(data_dict)), f, indent=2)
    return result


def ensure_fold_indices(cfg, data_dict):
    """Rebuild fold_indices.json when missing or config meta mismatches."""
    json_path = os.path.join(cfg.checkpoint_dir, "fold_indices.json")
    meta_path = os.path.join(cfg.checkpoint_dir, "fold_indices.meta.json")
    expected = _fold_meta(cfg, len(data_dict))
    if os.path.exists(json_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        if old == expected:
            return json_path
        print(f"[Split] fold_indices meta mismatch ({old} != {expected}); rebuilding")
    elif os.path.exists(json_path) and not os.path.exists(meta_path):
        print("[Split] fold_indices.meta.json missing; rebuilding for current config")
    build_fold_indices(cfg, data_dict)
    return json_path


def load_fold_indices(json_path, fold):
    with open(json_path, "r", encoding="utf-8") as f:
        all_indices = json.load(f)
    d = all_indices[str(fold)]
    return d["train_idx"], d["val_idx"], d["test_idx"]


def get_dataloaders(cfg, full_ds, fold):
    train_idx, val_idx, test_idx = load_fold_indices(
        os.path.join(cfg.checkpoint_dir, "fold_indices.json"), fold
    )
    image_keys = []
    if bool(getattr(cfg, "use_mri", True)):
        image_keys.append("MRI")
    if bool(getattr(cfg, "use_pet", True)):
        image_keys.append("PET")
    tf_tr, tf_val = ADNI_transform(
        augment=bool(getattr(cfg, "augment", False)),
        noise_std=float(getattr(cfg, "noise_std", 0.1)),
        noise_prob=float(getattr(cfg, "noise_prob", 0.5)),
        keys=image_keys,
    )
    ds_train = MonaiDataset([full_ds[i] for i in train_idx], transform=tf_tr)
    ds_val = MonaiDataset([full_ds[i] for i in val_idx], transform=tf_val)
    ds_test = MonaiDataset([full_ds[i] for i in test_idx], transform=tf_val)
    loader_tr = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    loader_te = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return loader_tr, loader_val, loader_te


def load_table_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c not in ("Subject_ID", "label")]
    dim = len(cols)
    mapping = {
        str(row["Subject_ID"]): row[cols].astype(np.float32).values
        for _, row in df.iterrows()
    }
    return mapping, dim


def resolve_table_map(cfg, full_ds, train_idx, fold):
    """Prefer tabular_emb.csv if present; else fold-wise TabularPreprocessor on table_dir."""
    emb_path = getattr(cfg, "tabular_emb", None)
    if emb_path not in (None, "", "null") and os.path.isfile(str(emb_path)):
        print(f"[Tabular] using precomputed embeddings: {emb_path}")
        return load_table_embeddings(str(emb_path))

    table_path = getattr(cfg, "table_dir", None) or getattr(cfg, "table_file", None)
    if not table_path or not os.path.isfile(table_path):
        raise FileNotFoundError(
            "No tabular features found. Provide an existing tabular_emb CSV "
            "or table_dir / table_file with baseline clinical variables."
        )
    table_df = load_tabular_csv(table_path)
    train_sids = [str(full_ds[i]["Subject"]) for i in train_idx]
    all_sids = [str(d["Subject"]) for d in full_ds]
    save_path = os.path.join(cfg.checkpoint_dir, f"tabular_preproc_fold{fold}.joblib")
    mapping, dim, _ = fit_transform_fold_table(
        table_df,
        train_sids,
        all_sids,
        task=cfg.task,
        feature_schema=getattr(cfg, "feature_schema", "full133"),
        scale_numeric=True,
        save_path=save_path,
    )
    print(
        f"[Tabular] fold {fold}: fit on train only -> dim={dim}, "
        f"schema={getattr(cfg, 'feature_schema', 'full133')}"
    )
    return mapping, dim


def get_table_tensor(subject_ids, table_map, tab_dim, device, batch=None):
    if batch is not None and "tabular" in batch:
        return batch["tabular"].to(device).float()
    vecs = []
    for sid in subject_ids:
        v = table_map.get(str(sid))
        if v is None:
            v = np.zeros((tab_dim,), dtype=np.float32)
        vecs.append(v)
    arr = np.stack(vecs, axis=0)
    return torch.from_numpy(arr).to(device)


def build_model(cfg, tab_dim: int) -> BRCMMHF:
    model = build_brc_mmhf_from_cfg(cfg, tab_dim=tab_dim)
    return model.to(cfg.device)


def build_optimizer(cfg, model):
    name = str(getattr(cfg, "optimizer", "adam")).lower()
    if name != "adam":
        raise ValueError(f"Unsupported optimizer '{name}'. The paper setting is 'adam'.")
    return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _batch_modalities(batch, cfg, table_map, tab_dim, device):
    use_mri = bool(getattr(cfg, "use_mri", True))
    use_pet = bool(getattr(cfg, "use_pet", True))
    use_table = bool(getattr(cfg, "use_table", True))

    mri = batch["MRI"].to(device) if use_mri else None
    pet = batch["PET"].to(device) if use_pet else None
    label = batch["label"].to(device).long()
    subjects = batch.get("Subject", [""] * label.size(0))
    table = None
    if use_table:
        table = get_table_tensor(subjects, table_map, tab_dim, device, batch=batch).float()
    return mri, pet, table, label, subjects


def epoch_run(loader, model, optimizer, scaler, criterion, device, table_map, tab_dim, fp16, cfg, desc=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    y_true_all, y_pred_all, y_prob_all = [], [], []

    seg_criterion = nn.BCEWithLogitsLoss()
    use_seg = bool(getattr(cfg, "use_seg_task", getattr(cfg, "seg_task", False)))
    seg_alpha = float(getattr(cfg, "seg_alpha", 0.05))
    use_mri = bool(getattr(cfg, "use_mri", True))
    use_pet = bool(getattr(cfg, "use_pet", True))
    seen_samples = 0

    progress = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        dynamic_ncols=True,
        leave=False,
        disable=not bool(getattr(cfg, "show_batch_progress", True)),
    )
    for batch_idx, batch in enumerate(progress, start=1):
        mri, pet, table, label, _ = _batch_modalities(batch, cfg, table_map, tab_dim, device)

        seg_mask = batch.get("seg_mask")
        if seg_mask is None and use_seg and use_mri and mri is not None:
            seg_mask = (mri > 0).float()
        elif seg_mask is not None:
            seg_mask = seg_mask.to(device).float()
            if seg_mask.ndim == 4:
                seg_mask = seg_mask.unsqueeze(1)

        with torch.set_grad_enabled(is_train):
            with autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=bool(fp16 and torch.cuda.is_available()),
            ):
                outputs = model(mri=mri, pet=pet, tabular=table)
                logits = outputs["logits"]
                cls_loss = criterion(logits, label)
                loss = cls_loss

                if use_seg and seg_mask is not None:
                    if seg_mask.ndim == 4:
                        seg_mask = seg_mask.unsqueeze(1)
                    seg_terms = []
                    seg_m = outputs.get("mri_seg")
                    seg_p = outputs.get("pet_seg")
                    if use_mri and seg_m is not None:
                        if seg_m.shape[1] != 1:
                            seg_m = seg_m[:, 1:2]
                        seg_terms.append(seg_criterion(seg_m, seg_mask))
                    if use_pet and seg_p is not None:
                        if seg_p.shape[1] != 1:
                            seg_p = seg_p[:, 1:2]
                        # PET may not share MRI mask intensity; still use MRI-derived brain mask as proxy
                        seg_terms.append(seg_criterion(seg_p, seg_mask))
                    if seg_terms:
                        loss = cls_loss + seg_alpha * (sum(seg_terms) / len(seg_terms))

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and bool(fp16):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1].float()
        preds = probs > 0.5
        loss_sum += loss.item() * label.size(0)
        seen_samples += label.size(0)
        y_true_all.extend(label.detach().cpu().numpy().tolist())
        y_pred_all.extend(preds.detach().cpu().numpy().astype(int).tolist())
        y_prob_all.extend(probs.detach().cpu().numpy().tolist())
        running_loss = loss_sum / max(seen_samples, 1)
        progress.set_postfix(loss=f"{running_loss:.4f}")

    epoch_loss = loss_sum / len(loader.dataset)
    metrics = calculate_metrics(y_true_all, y_pred_all, y_prob_all)
    return epoch_loss, metrics


def _fmt_metric(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.4f}"


def _fmt_epoch_metrics(split_name, loss, metrics):
    keys = ["ACC", "PRE", "SEN", "SPE", "BACC", "F1", "AUC", "MCC"]
    values = " ".join(f"{k}={_fmt_metric(metrics[k])}" for k in keys)
    return f"{split_name:<5} Loss={loss:.6f} {values}"


def train_fold(cfg, fold, loaders, table_map, tab_dim):
    device = cfg.device
    model = build_model(cfg, tab_dim=tab_dim)
    optimizer = build_optimizer(cfg, model)

    train_labels = [d["label"] for d in loaders[0].dataset.data]
    nb_class = int(getattr(cfg, "nb_class", 2))
    class_weights = compute_class_weights(train_labels, num_classes=nb_class).to(device)
    loss_name = getattr(cfg, "loss", "weighted_bce")
    criterion = make_loss(loss_name, device, num_classes=nb_class, class_weights=class_weights)
    scaler = GradScaler("cuda", enabled=bool(cfg.fp16 and torch.cuda.is_available()))
    best_auc = -np.inf
    best_path = os.path.join(cfg.checkpoint_dir, f"best_model_fold{fold}.pth")
    csv_path = os.path.join(cfg.checkpoint_dir, f"fold{fold}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "set", "Loss", "ACC", "PRE", "SEN", "SPE", "BACC", "F1", "AUC", "MCC"])

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_metrics = epoch_run(
            loaders[0], model, optimizer, scaler, criterion, device, table_map, tab_dim, cfg.fp16, cfg,
            desc=f"Fold {fold} Epoch {epoch:03d} train",
        )
        vl_loss, vl_metrics = epoch_run(
            loaders[1], model, None, None, criterion, device, table_map, tab_dim, cfg.fp16, cfg,
            desc=f"Fold {fold} Epoch {epoch:03d} val",
        )
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for split_name, loss, m in (("train", tr_loss, tr_metrics), ("val", vl_loss, vl_metrics)):
                w.writerow([
                    epoch, split_name, f"{loss:.6f}",
                    _fmt_metric(m["ACC"]), _fmt_metric(m["PRE"]), _fmt_metric(m["SEN"]),
                    _fmt_metric(m["SPE"]), _fmt_metric(m["BACC"]), _fmt_metric(m["F1"]),
                    _fmt_metric(m["AUC"]), _fmt_metric(m["MCC"]),
                ])
        score = vl_metrics["BACC"] if np.isnan(vl_metrics["AUC"]) else vl_metrics["AUC"]
        improved = score > best_auc
        if score > best_auc:
            best_auc = score
            torch.save({"model": model.state_dict(), "tab_dim": tab_dim}, best_path)
        t1 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        best_mark = " *best*" if improved else ""
        print(f"\nFold {fold} Epoch {epoch:03d}/{cfg.num_epochs} | {t1 - t0:.1f}s | lr={lr:.2e}{best_mark}", flush=True)
        print(_fmt_epoch_metrics("train", tr_loss, tr_metrics), flush=True)
        print(_fmt_epoch_metrics("val", vl_loss, vl_metrics), flush=True)
    return model, best_path, best_auc


@torch.no_grad()
def infer_fold(cfg, fold, test_loader, table_map, tab_dim, ckpt_path):
    device = cfg.device
    model = build_model(cfg, tab_dim=tab_dim)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    y_true, y_pred, y_prob = [], [], []
    for batch in test_loader:
        mri, pet, table, label, _ = _batch_modalities(batch, cfg, table_map, tab_dim, device)
        with autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=bool(cfg.fp16 and torch.cuda.is_available()),
        ):
            outputs = model(mri=mri, pet=pet, tabular=table)
            logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)[:, 1].float()
        preds = (probs > 0.5).int()
        y_true.extend(label.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    return metrics, y_true, y_pred, y_prob


def run_training(cfg, folds=None):
    """Reusable entry used by MMHF __main__ and experiments/run_ablation.py."""
    full_dataset = ADNI(cfg.label_file, cfg.mri_dir, cfg.pet_dir, cfg.task, cfg.augment)
    full_ds = full_dataset.data_dict
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ensure_fold_indices(cfg, full_ds)
    indices_path = os.path.join(cfg.checkpoint_dir, "fold_indices.json")
    results_txt = os.path.join(cfg.checkpoint_dir, "test_results.txt")
    result_csv = os.path.join(cfg.checkpoint_dir, "result.csv")

    with open(results_txt, "w", encoding="utf-8") as f:
        f.write("Fold\tACC\tPRE\tSEN\tSPE\tBACC\tF1\tAUC\tMCC\n")
    with open(result_csv, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow([
            "fold", "idx_in_fold", "sample_id", "true_label", "pred_label", "prob_pmci", "correct"
        ])

    if folds is None:
        folds = list(range(1, int(cfg.n_splits) + 1))
    all_metrics = []
    for fold in folds:
        print(f"=== Fold {fold}/{cfg.n_splits} ===")
        train_idx, _, _ = load_fold_indices(indices_path, fold)
        table_map, tab_dim = resolve_table_map(cfg, full_ds, train_idx, fold)
        loader_tr, loader_val, loader_te = get_dataloaders(cfg, full_ds, fold)
        _, best_path, best_auc = train_fold(
            cfg, fold, (loader_tr, loader_val, loader_te), table_map, tab_dim
        )
        metrics, y_true, y_pred, y_prob = infer_fold(
            cfg, fold, loader_te, table_map, tab_dim, best_path
        )
        all_metrics.append(metrics)
        with open(results_txt, "a", encoding="utf-8") as f:
            f.write(
                f"{fold}\t{metrics['ACC']:.4f}\t{metrics['PRE']:.4f}\t{metrics['SEN']:.4f}\t"
                f"{metrics['SPE']:.4f}\t{metrics['BACC']:.4f}\t{metrics['F1']:.4f}\t"
                f"{metrics['AUC']:.4f}\t{metrics['MCC']:.4f}\n"
            )
        with open(result_csv, "a", newline="", encoding="utf-8") as csv_f:
            writer = csv.writer(csv_f)
            test_data = loader_te.dataset.data
            for idx, sample_dict in enumerate(test_data):
                sid = sample_dict.get("Subject") or os.path.basename(
                    sample_dict.get("MRI", f"s{idx}")
                )
                writer.writerow([
                    fold, idx, sid, int(y_true[idx]), int(y_pred[idx]),
                    float(y_prob[idx]), int(y_true[idx] == y_pred[idx]),
                ])
        print(
            f"Fold {fold} done | best={best_auc:.4f} | "
            f"test AUC={metrics['AUC']:.4f} BACC={metrics['BACC']:.4f}"
        )

    print("=== Summary ===")
    keys = ["ACC", "PRE", "SEN", "SPE", "BACC", "F1", "AUC", "MCC"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
        if not vals:
            print(f"{k}: nan")
            summary[k] = (float("nan"), float("nan"))
        else:
            mean, std = float(np.mean(vals)), float(np.std(vals))
            print(f"{k}: {mean:.4f} ± {std:.4f}")
            summary[k] = (mean, std)
    return all_metrics, summary


def main():
    env_cfg = os.environ.get("MMHF_CONFIG")
    candidates = []
    if env_cfg and os.path.exists(env_cfg):
        candidates.append(env_cfg)
    candidates += [os.path.join("config", "config2.json"),
                   os.path.join("config", "config.json")]
    chosen = None
    cfg = None
    for p in candidates:
        if not os.path.exists(p):
            print(f"[Config] Skipping {p}: file not found")
            continue
        cfg_try = Cfg(load_cfg(p))
        if os.path.exists(cfg_try.label_file):
            chosen = p
            cfg = cfg_try
            print(f"[Config] Using {p}")
            break
        print(f"[Config] Skipping {p}: label_file '{cfg_try.label_file}' not found")
    if chosen is None or cfg is None:
        raise FileNotFoundError(
            "Valid configuration file path not found: Please set MMHF_CONFIG to point to a "
            "configuration containing an accessible label_file, or correct the data paths in config/config*.json"
        )
    # Ensure main training always targets the configured GPU (default cuda:0)
    cfg.device = resolve_device(getattr(cfg, "device", "cuda:0"))
    run_training(cfg)


if __name__ == "__main__":
    main()
