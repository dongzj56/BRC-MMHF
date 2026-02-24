import os, json, time, csv, numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from monai.data import Dataset as MonaiDataset
from sklearn.model_selection import StratifiedKFold
from utils.metrics import calculate_metrics
from datasets.ADNI import ADNI, ADNI_transform
from models.mmad_encoder import ImageEncoder_CEN, MultiModalClassifier


# Configuration load
def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Cfg:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        self.device = torch.device(self.device if torch.cuda.is_available() and str(self.device).startswith("cuda") else "cpu")


# Data Load
def build_fold_indices(cfg, data_dict):
    labels = [d["label"] for d in data_dict]
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    result = {}
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(np.arange(len(data_dict)), labels), start=1):
        train_val_labels = [labels[i] for i in train_val_idx]
        n_train = int(len(train_val_idx) * 0.9)
        train_idx = train_val_idx[:n_train].tolist()
        val_idx = train_val_idx[n_train:].tolist()
        result[str(fold_idx)] = {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx.tolist(),
        }
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    json_path = os.path.join(cfg.checkpoint_dir, "fold_indices.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def load_fold_indices(json_path, fold):
    with open(json_path, "r", encoding="utf-8") as f:
        all_indices = json.load(f)
    d = all_indices[str(fold)]
    return d["train_idx"], d["val_idx"], d["test_idx"]


# DataLoader Build
def get_dataloaders(cfg, full_ds, fold):
    train_idx, val_idx, test_idx = load_fold_indices(os.path.join(cfg.checkpoint_dir, "fold_indices.json"), fold)
    tf_tr, tf_val = ADNI_transform(augment=cfg.augment)
    tf_te = tf_val
    ds_train = MonaiDataset([full_ds[i] for i in train_idx], transform=tf_tr)
    ds_val = MonaiDataset([full_ds[i] for i in val_idx], transform=tf_val)
    ds_test = MonaiDataset([full_ds[i] for i in test_idx], transform=tf_te)
    loader_tr = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    loader_te = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return loader_tr, loader_val, loader_te


# Table embedding loading and mapping
def load_table_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c not in ("Subject_ID", "label")]
    dim = len(cols)
    mapping = {row["Subject_ID"]: row[cols].astype(np.float32).values for _, row in df.iterrows()}
    return mapping, dim


def get_table_tensor(subject_ids, table_map, tab_dim, device):
    vecs = []
    for sid in subject_ids:
        v = table_map.get(sid)
        if v is None:
            v = np.zeros((tab_dim,), dtype=np.float32)
        vecs.append(v)
    arr = np.stack(vecs, axis=0)
    return torch.from_numpy(arr).to(device)


# Model creation: Image encoder and multimodal classifier
def generate_image_model(cfg):
    model = ImageEncoder_CEN(
        in_ch_modality=1,
        level_channels=[64, 128, 256],
        bottleneck_ch=512,
        share_layers=2,
        cen_ratios=(0.20, 0.10),
    ).to(cfg.device)
    return model


def generate_mm_classifier(cfg, img_dim, tab_dim):
    model = MultiModalClassifier(
        img_dim=img_dim,
        tab_dim=tab_dim,
        num_classes=cfg.nb_class,
    ).to(cfg.device)
    return model


# ---- 单轮训练/验证 ----------------------------------------------------------
def epoch_run(loader, img_encoder, clf_model, optimizer, scaler, criterion, device, table_map, tab_dim, fp16):
    is_train = optimizer is not None
    if is_train:
        clf_model.train()
        img_encoder.eval()
    else:
        clf_model.eval()
        img_encoder.eval()
    loss_sum = 0.0
    y_true_all, y_pred_all, y_prob_all = [], [], []
    for batch in loader:
        mri = batch["MRI"].to(device)
        pet = batch["PET"].to(device)
        label = batch["label"].to(device).long()
        subjects = batch.get("Subject", [""] * mri.size(0))
        table = get_table_tensor(subjects, table_map, tab_dim, device).float()
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=bool(fp16)):
            img_feat = img_encoder(mri, pet)
            logits = clf_model(img_feat, table)
            loss = criterion(logits, label)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and bool(fp16):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = probs > 0.5
        loss_sum += loss.item() * label.size(0)
        y_true_all.extend(label.detach().cpu().numpy().tolist())
        y_pred_all.extend(preds.detach().cpu().numpy().astype(int).tolist())
        y_prob_all.extend(probs.detach().cpu().numpy().tolist())
    epoch_loss = loss_sum / len(loader.dataset)
    metrics = calculate_metrics(y_true_all, y_pred_all, y_prob_all)
    return epoch_loss, metrics


# ---- 单折训练：记录指标并保存最佳检查点 --------------------------------------
def train_fold(cfg, fold, loaders, table_map, tab_dim):
    device = cfg.device
    img_encoder = generate_image_model(cfg)
    clf_model = generate_mm_classifier(cfg, img_dim=1024, tab_dim=tab_dim)
    optimizer = torch.optim.AdamW(clf_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=bool(cfg.fp16))
    best_auc = -np.inf
    best_path = os.path.join(cfg.checkpoint_dir, f"best_model_fold{fold}.pth")
    csv_path = os.path.join(cfg.checkpoint_dir, f"fold{fold}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "set", "Loss", "ACC", "SEN", "SPE", "F1", "AUC", "MCC"])
    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_metrics = epoch_run(loaders[0], img_encoder, clf_model, optimizer, scaler, criterion, device, table_map, tab_dim, cfg.fp16)
        vl_loss, vl_metrics = epoch_run(loaders[1], img_encoder, clf_model, None, None, criterion, device, table_map, tab_dim, cfg.fp16)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, "train", f"{tr_loss:.6f}", f"{tr_metrics['ACC']:.4f}", f"{tr_metrics['SEN']:.4f}", f"{tr_metrics['SPE']:.4f}", f"{tr_metrics['F1']:.4f}", f"{tr_metrics['AUC']:.4f}", f"{tr_metrics['MCC']:.4f}"])
            w.writerow([epoch, "val", f"{vl_loss:.6f}", f"{vl_metrics['ACC']:.4f}", f"{vl_metrics['SEN']:.4f}", f"{vl_metrics['SPE']:.4f}", f"{vl_metrics['F1']:.4f}", f"{vl_metrics['AUC']:.4f}", f"{vl_metrics['MCC']:.4f}"])
        if vl_metrics["AUC"] > best_auc:
            best_auc = vl_metrics["AUC"]
            torch.save({"img_encoder": img_encoder.state_dict(), "clf_model": clf_model.state_dict()}, best_path)
        t1 = time.time()
        print(f"Fold {fold} Epoch {epoch} | train AUC={tr_metrics['AUC']:.4f} val AUC={vl_metrics['AUC']:.4f} | {t1 - t0:.1f}s")
    return {"img_encoder": img_encoder, "clf_model": clf_model}, best_path, best_auc


# ---- 单折推理：加载最佳权重并评估 -------------------------------------------
@torch.no_grad()
def infer_fold(cfg, fold, test_loader, table_map, tab_dim, ckpt_path):
    device = cfg.device
    img_encoder = generate_image_model(cfg)
    clf_model = generate_mm_classifier(cfg, img_dim=1024, tab_dim=tab_dim)
    ckpt = torch.load(ckpt_path, map_location=device)
    img_encoder.load_state_dict(ckpt["img_encoder"])
    clf_model.load_state_dict(ckpt["clf_model"])
    img_encoder.to(device).eval()
    clf_model.to(device).eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in test_loader:
        mri = batch["MRI"].to(device)
        pet = batch["PET"].to(device)
        label = batch["label"].to(device).long()
        subjects = batch.get("Subject", [""] * mri.size(0))
        table = get_table_tensor(subjects, table_map, tab_dim, device).float()
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=bool(cfg.fp16)):
            img_feat = img_encoder(mri, pet)
            logits = clf_model(img_feat, table)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > 0.5).int()
        y_true.extend(label.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    return metrics, y_true, y_pred, y_prob


# ---- 主流程：配置加载 → 划分 → 训练 → 测试 → 汇总 ---------------------------
def main():
    config_path = os.environ.get("MMHF_CONFIG", os.path.join("config", "config2.json"))
    cfg = Cfg(load_cfg(config_path))
    full_dataset = ADNI(cfg.label_file, cfg.mri_dir, cfg.pet_dir, cfg.task, cfg.augment)
    full_ds = full_dataset.data_dict
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    indices_path = os.path.join(cfg.checkpoint_dir, "fold_indices.json")
    if not os.path.exists(indices_path):
        build_fold_indices(cfg, full_ds)
    table_map, tab_dim = load_table_embeddings(cfg.tabular_emb)
    results_txt = os.path.join(cfg.checkpoint_dir, "test_results.txt")
    result_csv = os.path.join(cfg.checkpoint_dir, "result.csv")
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write("Fold\tACC\tPRE\tSEN\tSPE\tF1\tAUC\tMCC\n")
    with open(result_csv, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["fold", "idx_in_fold", "sample_id", "true_label", "pred_label", "correct"])
    all_metrics = []
    for fold in range(1, cfg.n_splits + 1):
        print(f"=== Fold {fold}/{cfg.n_splits} ===")
        loader_tr, loader_val, loader_te = get_dataloaders(cfg, full_ds, fold)
        models, best_path, best_auc = train_fold(cfg, fold, (loader_tr, loader_val, loader_te), table_map, tab_dim)
        metrics, y_true, y_pred, y_prob = infer_fold(cfg, fold, loader_te, table_map, tab_dim, best_path)
        all_metrics.append(metrics)
        with open(results_txt, "a", encoding="utf-8") as f:
            f.write(f"{fold}\t{metrics['ACC']:.4f}\t{metrics['PRE']:.4f}\t{metrics['SEN']:.4f}\t{metrics['SPE']:.4f}\t{metrics['F1']:.4f}\t{metrics['AUC']:.4f}\t{metrics['MCC']:.4f}\n")
        with open(result_csv, "a", newline="", encoding="utf-8") as csv_f:
            writer = csv.writer(csv_f)
            test_data = loader_te.dataset.data
            for idx, sample_dict in enumerate(test_data):
                sid = sample_dict.get("Subject") or os.path.basename(sample_dict.get("MRI", f"s{idx}"))
                writer.writerow([fold, idx, sid, int(y_true[idx]), int(y_pred[idx]), int(y_true[idx] == y_pred[idx])])
        print(f"Fold {fold} done | best AUC={best_auc:.4f} | test AUC={metrics['AUC']:.4f}")
    print("=== Summary ===")
    keys = ["ACC", "PRE", "SEN", "SPE", "F1", "AUC", "MCC"]
    for k in keys:
        vals = [m[k] for m in all_metrics]
        print(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
