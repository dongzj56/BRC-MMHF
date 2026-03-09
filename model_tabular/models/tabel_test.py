# This script generates embeddings for ADNI tabular data using a PFN-style encoder.
import os, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tabpfn")
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding
from datasets.tabel_loader import load_adni_data_binary

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"▶ Using device: {DEVICE}")

def tabel_encoder(
    csv_path: str,
    start_col: int = 14,
    class0: str = "AD",
    class1: str = "CN",
    n_fold: int = 5,
    test_size: float = 0.3,
    random_state: int = 42,
    train_out: str = "train_embeddings.csv",
    test_out: str = "test_embeddings.csv"
):
    X, y = load_adni_data_binary(
        csv_path,
        start_col=start_col,
        class0=class0,
        class1=class1
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    clf = TabPFNClassifier(device=device)
    embed = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)

    train_emb = embed.get_embeddings(X_tr, y_tr, X_te, data_source="train")[0]
    test_emb  = embed.get_embeddings(X_tr, y_tr, X_te, data_source="test")[0]

    train_df = pd.DataFrame(train_emb)
    train_df.insert(0, "label", y_tr)
    train_df.to_csv(train_out, index=False)

    test_df = pd.DataFrame(test_emb)
    test_df.insert(0, "label", y_te)
    test_df.to_csv(test_out, index=False)

    print(f"Train embeddings saved to: {train_out}")
    print(f"Test embeddings saved to: {test_out}")

def tabel_encoder_multi(
    csv_path: str,
    start_col: int=14,
    label_col: str="GROUP",
    classes: list=["CN", "AD"],
    n_fold: int = 5,
    test_size: float = 0.3,
    random_state: int = 42,
    train_out: str = "train_embeddings.csv",
    test_out: str = "test_embeddings.csv"
):
    df = pd.read_csv(csv_path, dtype={label_col: str})
    df = df[df[label_col].isin(classes)]
    if df.empty:
        raise ValueError(f"Error: No labels found in {csv_path} belonging to {classes}.")

    X = df.iloc[:, start_col:].values
    y_str = df[label_col].values
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    y_num = pd.Series(y_str).map(label_to_index).values

    X_tr, X_te, y_tr_num, y_te_num, y_tr_str, y_te_str = train_test_split(
        X, y_num, y_str, test_size=test_size, random_state=random_state, stratify=y_num
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    clf = TabPFNClassifier(device=device)
    embed = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)

    train_emb = embed.get_embeddings(X_tr, y_tr_num, X_te, data_source="train")[0]
    test_emb  = embed.get_embeddings(X_tr, y_tr_num, X_te, data_source="test")[0]
    print(f"train_emb shape: {train_emb.shape}")
    print(f"test_emb shape: {test_emb.shape}")

    train_df = pd.DataFrame(train_emb)
    train_df.insert(0, "label", y_tr_str)
    train_df.to_csv(train_out, index=False)

    test_df = pd.DataFrame(test_emb)
    test_df.insert(0, "label", y_te_str)
    test_df.to_csv(test_out, index=False)

    print(f"Saved {len(y_tr_str)} training embeddings (with labels) to: {train_out}")
    print(f"Saved {len(y_te_str)} testing embeddings (with labels) to: {test_out}")

def quick_eval_from_saved(train_csv="train_embeddings.csv", test_csv="test_embeddings.csv"):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)

    y_tr, X_tr = tr["label"].values, tr.drop(columns="label").values
    y_te, X_te = te["label"].values, te.drop(columns="label").values

    clf = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"[quick eval · SVM-linear] Accuracy on {test_csv}: {acc:.4f}")
    return acc

if __name__ == "__main__":
    print("embedings.......")
    tabel_encoder(csv_path=rf'C:\Users\dongzj\Desktop\Multimodal_AD\adni_dataset\ADNI_Tabel.csv')
    print("test model......")
    quick_eval_from_saved()
