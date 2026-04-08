"""
=============================================================================
MASTER MACHINE LEARNING & EVALUATION PIPELINE
=============================================================================
CONTRIBUTORS:
1. Kasak Singh      -> Logistic Regression & K-Nearest Neighbors
2. Kavyansh Vats    -> Decision Tree Classifier
3. Tanishi Rai      -> Naive Bayes Classifier
4. Sanjiban Ghosh   -> K-Means Clustering & Statistical Evaluation Suite
5. Ronak Kamboj     -> Data Visualization (Confusion Matrices & Plots)
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
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

    # Dictionary to store predictions for Ronak's visualization phase
    model_predictions = {}

    # =========================================================
    # PART 1: SUPERVISED LEARNING CLASSIFIERS
    # =========================================================
    
    print("\n--- Training Logistic Regression (Kasak) ---")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    lr_preds = log_reg.predict(X_test)
    model_predictions['Logistic Regression'] = lr_preds
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_preds):.4f}")

    print("\n--- Training Decision Tree (Kavyansh) ---")
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    dt_preds = dt_clf.predict(X_test)
    model_predictions['Decision Tree'] = dt_preds
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_preds):.4f}")

    print("\n--- Training KNN on 20k subset (Kasak) ---")
    X_train_knn = X_train.sample(n=20000, random_state=42)
    y_train_knn = y_train.loc[X_train_knn.index]
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_knn, y_train_knn)
    knn_preds = knn.predict(X_test)
    model_predictions['KNN'] = knn_preds
    print(f"KNN Accuracy: {accuracy_score(y_test, knn_preds):.4f}")

    print("\n--- Training Naive Bayes (Tanishi) ---")
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    nb_preds = nb_clf.predict(X_test)
    model_predictions['Naive Bayes'] = nb_preds
    print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_preds):.4f}")

    # =========================================================
    # PART 2: UNSUPERVISED LEARNING
    # =========================================================
    
    print("\n--- Training K-Means Clustering (Sanjiban) ---")
    # K-Means is heavy, subsample for speed. 14 clusters for 14 labels.
    X_train_kmeans = X_train.sample(n=50000, random_state=42)
    kmeans = KMeans(n_clusters=14, random_state=42, n_init='auto')
    kmeans.fit(X_train_kmeans)
    
    # Predict clusters for a small subset of test data for Ronak's visualization
    X_test_viz = X_test.sample(n=5000, random_state=42)
    kmeans_clusters = kmeans.predict(X_test_viz)
    print(f"K-Means Inertia: {kmeans.inertia_:.2f}")


    # =========================================================
    # PART 3: STATISTICAL EVALUATION SUITE (Sanjiban)
    # =========================================================
    
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
    print(f"Saved ROC curve to {roc_path}")


    # =========================================================
    # PART 4: DATA VISUALIZATION (Ronak)
    # =========================================================
    print("\n--- Generating Model Visualizations ---")
    
    # 1. Confusion Matrices for LogReg, Decision Tree, Naive Bayes
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Classifier Confusion Matrices', fontsize=16, fontweight='bold')

    # We select just 3 models to fit nicely side-by-side
    viz_models = ['Logistic Regression', 'Decision Tree', 'Naive Bayes']
    for ax, model_name in zip(axes, viz_models):
        preds = model_predictions[model_name]
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=False, cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    cm_path = BASE_DIR / "confusion_matrices.png"
    plt.savefig(cm_path)
    print(f"Saved Confusion Matrices to: {cm_path}")

    # 2. K-Means Cluster Visualization
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_test_viz['PC1'], 
        y=X_test_viz['PC2'], 
        hue=kmeans_clusters, 
        palette='tab20', 
        alpha=0.7, 
        legend=False
    )
    plt.title('K-Means Clustering Distribution (PC1 vs PC2)', fontweight='bold')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    cluster_path = BASE_DIR / "kmeans_clusters.png"
    plt.savefig(cluster_path)
    print(f"Saved K-Means Cluster plot to: {cluster_path}")
    
    print("\n✅ All models trained and visual reports generated successfully!")

if __name__ == "__main__":
    train_and_evaluate()
