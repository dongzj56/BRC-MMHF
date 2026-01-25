import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression   # 也可换 XGBoost / SVM 等

# ────────────────────────────────────────────────
# 1) 读取向量与标签
# ────────────────────────────────────────────────
X_train = pd.read_csv("train_embeddings.csv").values   # shape: (n_train, d)
X_test  = pd.read_csv("test_embeddings.csv").values    # shape: (n_test,  d)
y_train = pd.read_csv("train_labels.csv").values.ravel()
y_test  = pd.read_csv("test_labels.csv").values.ravel()

print("train:", X_train.shape, " test:", X_test.shape)

# ────────────────────────────────────────────────
# 2) 直接训练分类器
# ────────────────────────────────────────────────
clf = LogisticRegression(max_iter=1000)   # 换成其它模型同理
clf.fit(X_train, y_train)

# ────────────────────────────────────────────────
# 3) 评估
# ────────────────────────────────────────────────
y_pred  = clf.predict(X_test)
y_prob  = clf.predict_proba(X_test)[:, 1]   # 若模型支持概率输出

print("Accuracy :", accuracy_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))
