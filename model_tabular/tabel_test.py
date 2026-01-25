"""tabel_test.py – ADNI TabPFN Embedding Pipeline
================================================
• 自动 CPU/GPU 切换
• 类别型列整数编码
• 支持 Vanilla 与 K‑Fold OoF 两种 TabPFN 嵌入
"""
import os, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tabpfn")
# os.environ["TABPFN_DISABLE_CPU_WARNING"] = "1"  # 完全屏蔽提示可取消注释
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
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
    """
    读取 ADNI 表格数据，生成 TabPFN 嵌入并保存到 CSV 文件。

    参数:
        csv_path (str): ADNI 表格文件路径 (例如 "ADNI_Tabel.csv")。
        start_col (int): 从第几列开始使用表格特征 (默认为 18)。
        class0 (str): 类别 0 对应的标签 (默认为 "SMCI")。
        class1 (str): 类别 1 对应的标签 (默认为 "PMCI")。
        n_fold (int): TabPFNEmbedding 的折数，若要使用 Vanilla 模式请传 0 (默认为 5)。
        test_size (float): 测试集比例 (默认为 0.3)。
        random_state (int): 随机种子 (默认为 42)。
        train_out (str): 训练集嵌入保存路径 (默认为 "train_embeddings.csv")。
        test_out (str): 测试集嵌入保存路径 (默认为 "test_embeddings.csv")。

    返回:
        None  (会将带标签的嵌入分别写入 train_out 和 test_out)
    """
    # 1. 读取表格并拆分标签
    X, y = load_adni_data_binary(
        csv_path,
        start_col=start_col,
        class0=class0,
        class1=class1
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. 确定设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 3. 初始化 TabPFNClassifier 与 TabPFNEmbedding
    clf = TabPFNClassifier(device=device)
    embed = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)

    # 4. 生成嵌入 (train_emb, test_emb 都是 numpy 数组，shape=(样本数, 嵌入维度))
    train_emb = embed.get_embeddings(X_tr, y_tr, X_te, data_source="train")[0]
    test_emb  = embed.get_embeddings(X_tr, y_tr, X_te, data_source="test")[0]

    # 5. 将嵌入包装为 DataFrame，并在第 0 列插入标签
    train_df = pd.DataFrame(train_emb)
    train_df.insert(0, "label", y_tr)
    train_df.to_csv(train_out, index=False)

    test_df = pd.DataFrame(test_emb)
    test_df.insert(0, "label", y_te)
    test_df.to_csv(test_out, index=False)

    print(f"已将训练集嵌入保存到：{train_out}")
    print(f"已将测试集嵌入保存到：{test_out}")

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
    """
    通用的多分类 tabel_encoder：支持 3 类或 4 类（或任意类别数，只要 classes 列表长度 >= 2）。

    参数:
        csv_path (str): 包含标签和特征的 CSV 文件路径。
        start_col (int): 从第几列开始作为特征（0-based 索引），
                         即 label_col 所在列在 start_col 之前。
        label_col (str): 标签列名，CSV 中存放类别字符串的那一列。
        classes (list of str): 要纳入的类别名称列表（例如 ["CN", "SMCI", "PMCI"] 三类，或 ["CN", "EMCI", "LMCI", "AD"] 四类）。
        n_fold (int): TabPFNEmbedding 使用的折数；若要 Vanilla （不划分折）请传 0。默认 5。
        test_size (float): 测试集比例，默认为 0.3。
        random_state (int): 随机种子以保证可复现。默认 42。
        train_out (str): 训练集嵌入保存路径。默认 "train_embeddings.csv"。
        test_out (str): 测试集嵌入保存路径。默认 "test_embeddings.csv"。

    整体流程：
      1. 读取 CSV 并过滤只保留 classes 列表中的行
      2. 提取特征矩阵 X (从 start_col 及之后的所有列)；
         提取标签列 y_str (原始字符串标签)，并映射成 y_num (0,1,2,…)
      3. 按照 test_size、random_state 划分训练/测试集，
         同时拆分 X, y_num, y_str 三者
      4. 用 TabPFNEmbedding 生成训练和测试数据的 embedding
      5. 将 embedding 转换成 DataFrame，并在首列插入原始标签字符串 y_str，
         最后保存成 CSV
    """
    # 1. 读取 CSV，将只属于 classes 列表的行过滤出来
    df = pd.read_csv(csv_path, dtype={label_col: str})
    df = df[df[label_col].isin(classes)]
    if df.empty:
        raise ValueError(f"——错误：在 {csv_path} 中未找到任何属于 {classes} 的标签。")

    # 2. 构建 X、y_str、y_num
    X = df.iloc[:, start_col:].values  # 从 start_col 开始后面的所有列作为特征
    y_str = df[label_col].values       # 原始标签字符串数组
    # 将字符串标签映射为 0,1,2,… 数字
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    y_num = pd.Series(y_str).map(label_to_index).values

    # 3. 同时拆分 X、y_num、y_str
    X_tr, X_te, y_tr_num, y_te_num, y_tr_str, y_te_str = train_test_split(
        X, y_num, y_str, test_size=test_size, random_state=random_state, stratify=y_num
    )

    # 4. 确定 TabPFNClassifier 运行设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 5. 初始化分类器和嵌入器
    clf = TabPFNClassifier(device=device)
    embed = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)

    # 6. 生成 train 和 test 的 embedding（都会返回元组，第 0 项是我们要的 embedding 数组）
    train_emb = embed.get_embeddings(X_tr, y_tr_num, X_te, data_source="train")[0]
    test_emb  = embed.get_embeddings(X_tr, y_tr_num, X_te, data_source="test")[0]
    print(f"train_emb shape: {train_emb.shape}")  # 输出如 (样本数, 嵌入维度)
    print(f"test_emb shape: {test_emb.shape}")

    # 7. 保存到 DataFrame，并在第 0 列插入原始标签字符串
    train_df = pd.DataFrame(train_emb)
    train_df.insert(0, "label", y_tr_str)
    train_df.to_csv(train_out, index=False)

    test_df = pd.DataFrame(test_emb)
    test_df.insert(0, "label", y_te_str)
    test_df.to_csv(test_out, index=False)

    print(f"✔ 已将 {len(y_tr_str)} 条训练样本的 embedding（含标签）保存到：{train_out}")
    print(f"✔ 已将 {len(y_te_str)} 条测试样本的 embedding（含标签）保存到：{test_out}")

# ----------------------------------------------------------------------
# 简易后处理：读取保存的文件直接做分类
def quick_eval_from_saved(train_csv="train_embeddings.csv", test_csv="test_embeddings.csv"):
    """
    读取带标签的嵌入 CSV，使用 **最简 SVM** (线性核) 做一次快速评估。
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)

    y_tr, X_tr = tr["label"].values, tr.drop(columns="label").values
    y_te, X_te = te["label"].values, te.drop(columns="label").values

    # 使用线性核 SVM, 外加标准化, 这是最简易且常用的组合
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"[quick eval · SVM-linear] Accuracy on {test_csv}: {acc:.4f}")
    return acc

# 若脚本作为主程序运行，则执行快速评估
if __name__ == "__main__":
    print("embedings.......")
    tabel_encoder(csv_path=rf'C:\Users\dongzj\Desktop\Multimodal_AD\adni_dataset\ADNI_Tabel.csv')
    # tabel_encoder_multi(csv_path='ADNI_Tabel.csv',label_col='GROUP',classes=["SCMI","PMCI","AD","CN"], n_fold=5,test_size= 0.2)
    print("test model......")
    quick_eval_from_saved()

