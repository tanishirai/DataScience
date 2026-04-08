"""
=============================================================================
CONTRIBUTOR 1: Kasak Singh (Machine Learning Setup & Training)
CONTRIBUTOR 2: Sanjiban Ghosh (Model Evaluation Suite)
DESCRIPTION: Splits data, trains classifiers (Logistic, Decision Tree, KNN), 
             and evaluates performance via Classification Reports, T-Tests, 
             and Multiclass ROC Curves.
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import stats
from config import PROCESSED_DATA_PATH, BASE_DIR

def train_and_evaluate():
    print(f"Loading processed data from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop('label', axis=1)
    y = df['label']

    print("Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Training Logistic Regression ---")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    lr_preds = log_reg.predict(X_test)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_preds):.4f}")

    print("\n--- Training Decision Tree ---")
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    dt_preds = dt_clf.predict(X_test)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_preds):.4f}")

    print("\n--- Training KNN (on 20k subset) ---")
    X_train_knn = X_train.sample(n=20000, random_state=42)
    y_train_knn = y_train.loc[X_train_knn.index]
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_knn, y_train_knn)
    knn_preds = knn.predict(X_test)
    print(f"KNN Accuracy: {accuracy_score(y_test, knn_preds):.4f}")

    print("\n--- Classification Report (Logistic Regression) ---")
    print(classification_report(y_test, lr_preds, zero_division=0))

    print("\n--- Model Comparison T-Test (LogReg vs Liblinear) ---")
    X_sample = X.sample(n=50000, random_state=42)
    y_sample = y.loc[X_sample.index]

    scores_1 = cross_val_score(LogisticRegression(max_iter=1000), X_sample, y_sample, cv=3)
    scores_2 = cross_val_score(LogisticRegression(solver='liblinear', max_iter=1000), X_sample, y_sample, cv=3)
    t_stat, p_val = stats.ttest_rel(scores_1, scores_2)
    
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Result: Statistically Significant difference.")
    else:
        print("Result: No statistically significant difference.")

    print("\n--- Generating Multiclass ROC Curve ---")
    classes = sorted(y.unique())
    y_test_bin = label_binarize(y_test, classes=classes)
    y_probs = log_reg.predict_proba(X_test)

    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_probs.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Micro-average ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Multiclass)')
    plt.legend(loc="lower right")
    
    roc_path = BASE_DIR / "roc_curve.png"
    plt.savefig(roc_path)
    print(f"ROC curve saved to {roc_path}")

if __name__ == "__main__":
    train_and_evaluate()
