"""
=============================================================================
CONTRIBUTOR: Tanishi Rai
SECTION: Data Integration & Cleaning
DESCRIPTION: Merges raw CSV files, handles missing values, removes duplicates, 
             and optimizes memory usage via data downcasting.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import gc
from config import RAW_DATA_DIR, INTERIM_DATA_PATH

def clean_data():
    print("1. Loading raw data...")
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    if not files:
        print("ERROR: No CSV files found in data/raw/. Please add them.")
        return

    dfs = []
    for file in files:
        print(f"   Loading: {file}")
        dfs.append(pd.read_csv(os.path.join(RAW_DATA_DIR, file)))
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Free up memory immediately after concatenation
    del dfs
    gc.collect()

    print(f"Combined shape: {df.shape}")

    print("2. Cleaning columns and dropping duplicates...")
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop_duplicates().dropna()

    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).str.strip()

    for col in df.columns:
        if col != 'label':
            df[col] = pd.to_numeric(df[col], errors='ignore')

    print("3. Downcasting memory footprint...")
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    print("4. Removing constant columns...")
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df = df.drop(columns=cols_to_drop)

    print("5. Handling negative values and outliers...")
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)

    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)

    print(f"Saving interim data to {INTERIM_DATA_PATH}...")
    df.to_csv(INTERIM_DATA_PATH, index=False)
    print("Cleaning Complete!")

if __name__ == "__main__":
    clean_data()
