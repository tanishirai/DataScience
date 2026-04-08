"""
=============================================================================
CONTRIBUTOR 1: Kavyansh Vats (Data Transformation & Reduction)
CONTRIBUTOR 2: Ronak Kamboj (Data Discretization)
DESCRIPTION: Applies Label Encoding, Standard Scaling, Principal Component 
             Analysis (PCA), and discretizes features into Low/Medium/High bins.
=============================================================================
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from config import INTERIM_DATA_PATH, PROCESSED_DATA_PATH

def preprocess_data():
    print(f"1. Loading interim data from {INTERIM_DATA_PATH}...")
    df = pd.read_csv(INTERIM_DATA_PATH)

    print("2. Encoding categorical labels...")
    label_enc = LabelEncoder()
    df['label_encoded'] = label_enc.fit_transform(df['label'])

    X = df.drop(columns=['label', 'label_encoded'])
    y = df['label_encoded']

    print("3. Standardizing features...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print("4. Applying PCA (95% variance)...")
    pca = PCA(n_components=0.95)
    pca_data = pca.fit_transform(X_scaled)
    
    pca_cols = [f'PC{i+1}' for i in range(pca_data.shape[1])]
    df_pca = pd.DataFrame(pca_data, columns=pca_cols)
    print(f"   Reduced to {pca_data.shape[1]} principal components.")

    print("5. Discretizing PCA features into Low/Medium/High...")
    df_final = df_pca.copy()
    for col in pca_cols:
        df_final[col] = pd.qcut(df_final[col], q=3, labels=[0, 1, 2], duplicates='drop')

    df_final['label'] = y.values

    print(f"Saving fully processed data to {PROCESSED_DATA_PATH}...")
    df_final.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Preprocessing Complete!")

if __name__ == "__main__":
    preprocess_data()
