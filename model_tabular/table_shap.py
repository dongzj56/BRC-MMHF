"""
tabel_shap.py  –  TabPFN  ➜  SHAP 可解释性分析
================================================
• 支持 Vanilla / K‑Fold 两种嵌入方式
• 输出 numpy SHAP 值和两张标准图
"""

import os, warnings, torch, shap, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding
from datasets.tabel_loader import load_adni_data_binary

warnings.filterwarnings("ignore", category=UserWarning, module="tabpfn")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"▶ Using device: {DEVICE}")

# --------------------------- 嵌入阶段 --------------------------- #
def tabel_encoder_binary(
    csv_path: str,
    start_col: int,
    class0: str,
    class1: str,
    n_fold: int,
    test_size: float,
    random_state: int = 42,
):
    """返回 (X_train, X_test, y_train, y_test, clf)；不再落盘 CSV"""
    X, y = load_adni_data_binary(csv_path, start_col, class0, class1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = TabPFNClassifier(device=DEVICE)
    embedder = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)

    # get_embeddings → Tuple(embeddings, fitted_classifier)
    tr_emb, _ = embedder.get_embeddings(X_tr, y_tr, X_te, data_source="train")
    te_emb, _ = embedder.get_embeddings(X_tr, y_tr, X_te, data_source="test")

    return tr_emb, te_emb, y_tr, y_te, clf

# --------------------------- SHAP 分析 --------------------------- #
def shap_analysis(
    clf: TabPFNClassifier,
    X_train_emb: np.ndarray,
    X_test_emb: np.ndarray,
    n_background: int = 100,
    out_prefix: str = "shap_tabpfn",
):
    """KernelExplainer + 保存 numpy & 图片"""
    # 1. 背景样本
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train_emb), size=min(n_background, len(X_train_emb)), replace=False)
    background = X_train_emb[bg_idx]

    # 2. 包装预测函数（正类概率）
    def model_predict(x):
        return clf.predict_proba(x)[:, 1]

    # 3. 创建 Explainer 并计算 SHAP
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(X_test_emb, nsamples="auto")
    np.save(f"{out_prefix}_values.npy", shap_values)
    print(f"✔ SHAP 值已保存到 {out_prefix}_values.npy，shape={shap_values.shape}")

    # 4. 绘图
    shap.summary_plot(shap_values, features=X_test_emb,
                      feature_names=[f"f{i}" for i in range(X_test_emb.shape[1])],
                      show=False, plot_type="bar")
    shap.plt.gcf().savefig(f"{out_prefix}_summary_bar.png", bbox_inches="tight")
    shap.summary_plot(shap_values, features=X_test_emb,
                      feature_names=[f"f{i}" for i in range(X_test_emb.shape[1])],
                      show=False)
    shap.plt.gcf().savefig(f"{out_prefix}_summary_beeswarm.png", bbox_inches="tight")
    print(f"✔ SHAP 图已保存到 {out_prefix}_summary_bar.png / _beeswarm.png")

# --------------------------- 主入口 --------------------------- #
if __name__ == "__main__":
    CSV = "ADNI_Tabel.csv"          # ← 你的原始表格
    START_COL = 20
    CLASS0, CLASS1 = "AD", "CN"     # 可改为 "SMCI","PMCI"
    N_FOLD = 5                      # 0=Vanilla ；>0=K‑Fold OoF
    TEST_SIZE = 0.2

    print("⏳ 生成 TabPFN 嵌入 ...")
    X_tr_emb, X_te_emb, y_tr, y_te, clf = tabel_encoder_binary(
        csv_path=CSV,
        start_col=START_COL,
        class0=CLASS0,
        class1=CLASS1,
        n_fold=N_FOLD,
        test_size=TEST_SIZE,
    )

    print("⏳ 计算 SHAP ...")
    shap_analysis(
        clf=clf,
        X_train_emb=X_tr_emb,
        X_test_emb=X_te_emb,
        out_prefix=f"shap_{CLASS0}_{CLASS1}"
    )

    print("🎉 完成全部流程")
