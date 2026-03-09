import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

# 1. Load Embeddings and Labels
X_train = pd.read_csv("train_embeddings.csv").values
X_test  = pd.read_csv("test_embeddings.csv").values
y_train = pd.read_csv("train_labels.csv").values.ravel()
y_test  = pd.read_csv("test_labels.csv").values.ravel()

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 2. Train Classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 3. Evaluate
y_pred  = clf.predict(X_test)
y_prob  = clf.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))
