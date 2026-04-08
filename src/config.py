import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
INTERIM_DATA_PATH = BASE_DIR / "data" / "interim" / "cleaned_data.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "final_data.csv"
