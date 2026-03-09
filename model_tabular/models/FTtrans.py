# ft_transformer_run.py
import os, random, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

CONFIG = {
    "csv":        r"adni_dataset/ADNI_Tabel.csv",
    "label_col":  "Group",
    "cat_cols":   "PTGENDER,APOE4",
    "num_cols":   "AGE,PTEDUCAT,MMSE",
    "test_size":  0.2,
    "batch_size": 128,
    "epochs":     30,
    "dim":        128,          # d_model
    "n_head":     8,
    "depth":      4,
    "lr":         2e-4,
    "seed":       42,
}

class FTEncoder(nn.Module):
    def __init__(self, cat_dims, num_dim, dim=128, n_head=8, depth=4, dropout=0.1):
        super().__init__()
        self.cat_offsets = np.cumsum([0] + cat_dims[:-1]).astype("int64")
        self.cat_embed   = nn.Embedding(sum(cat_dims), dim)
        self.num_proj    = nn.Linear(num_dim, dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_head, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, depth)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_cat, x_num):
        x_cat = x_cat + x_cat.new_tensor(self.cat_offsets)
        cat_tok = self.cat_embed(x_cat)
        num_tok = self.num_proj(x_num).unsqueeze(1)
        B = x_cat.size(0)
        x = torch.cat([self.cls_token.expand(B, -1, -1), cat_tok, num_tok], dim=1)
        x = self.transformer(x)
        return self.norm(x[:, 0])

class FTClassifier(nn.Module):
    def __init__(self, encoder, dim=128, n_cls=2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, n_cls)

    def forward(self, x_cat, x_num):
        h = self.encoder(x_cat, x_num)
        return self.head(h)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train(); total=correct=0
    for xc, xn, y in loader:
        xc, xn, y = xc.to(device), xn.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xc, xn), y)
        loss.backward(); optimizer.step()
        total += y.size(0); correct += (model(xc, xn).argmax(1)==y).sum().item()
    return correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); y_true=[]; y_prob=[]
    for xc,xn,y in loader:
        prob = torch.softmax(model(xc.to(device), xn.to(device)),1)[:,1].cpu().numpy()
        y_prob.append(prob); y_true.append(y.numpy())
    y_prob = np.concatenate(y_prob); y_true=np.concatenate(y_true)
    y_pred = (y_prob>=0.5).astype("int64")
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    return acc, auc, y_true, y_pred

def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Data
    df = pd.read_csv(cfg["csv"])
    cat_cols = cfg["cat_cols"].split(',')
    num_cols = cfg["num_cols"].split(',')

    df = df.dropna(subset=cat_cols+num_cols+[cfg["label_col"]]).copy()
    df[cfg["label_col"]] = pd.Categorical(df[cfg["label_col"]]).codes
    scaler = StandardScaler(); df[num_cols] = scaler.fit_transform(df[num_cols])
    cat_dims=[]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes
        cat_dims.append(df[col].max()+1)

    X_cat = df[cat_cols].values
    X_num = df[num_cols].values.astype("float32")
    y     = df[cfg["label_col"]].values

    Xc_tr,Xc_te,Xn_tr,Xn_te,y_tr,y_te = train_test_split(
        X_cat,X_num,y,test_size=cfg["test_size"],stratify=y,random_state=cfg["seed"])

    tr_loader = DataLoader(TensorDataset(torch.tensor(Xc_tr, dtype=torch.long),
              torch.tensor(Xn_tr),
              torch.tensor(y_tr)),
                           batch_size=cfg["batch_size"],shuffle=True)
    te_loader = DataLoader(TensorDataset(torch.tensor(Xc_te, dtype=torch.long),
              torch.tensor(Xn_te),
              torch.tensor(y_te)),
                           batch_size=cfg["batch_size"],shuffle=False)

    # 2. Model
    enc = FTEncoder(cat_dims, len(num_cols), cfg["dim"], cfg["n_head"], cfg["depth"]).to(device)
    model = FTClassifier(enc, cfg["dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    crit = nn.CrossEntropyLoss()

    best_acc=0
    for epoch in range(1, cfg["epochs"]+1):
        tr_acc = train_one_epoch(model, tr_loader, opt, crit, device)
        vl_acc, vl_auc, _, _ = evaluate(model, te_loader, device)
        print(f"[{epoch}/{cfg['epochs']}] TrainACC {tr_acc:.3f} | ValACC {vl_acc:.3f}  AUC {vl_auc:.3f}")
        if vl_acc>best_acc:
            best_acc=vl_acc; torch.save(model.state_dict(),"best_ftt.pth")

    model.load_state_dict(torch.load("best_ftt.pth"))
    acc, auc, y_true, y_pred = evaluate(model, te_loader, device)
    print(f"\n=== Final Test ===  ACC={acc:.4f}  AUC={auc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=["Class0","Class1"]))

if __name__ == "__main__":
    main(CONFIG)
