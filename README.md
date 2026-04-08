# Data Science Project – Local Setup & Pipeline Guide

## Project Overview
This project is an end-to-end machine learning pipeline designed to efficiently handle a dataset with over 2.3 million rows. The architecture has been transitioned from a monolithic Jupyter Notebook to a modular, production-ready local setup to improve memory efficiency, prevent RAM crashes, and enable better collaboration.

### Key Capabilities
- Data integration, cleaning, and memory optimization (downcasting)
- Feature engineering, scaling, PCA, and discretization
- Exploratory Data Analysis (EDA)
- Machine learning model training:
  - Logistic Regression  
  - Decision Trees  
  - K-Nearest Neighbors (KNN)
- Statistical evaluation:
  - Classification reports  
  - ROC curves  
  - Paired T-tests  

---

## Project Architecture

```
ds-project/
├── .gitignore                           # Ignores large data files 
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
│
├── data/                                # DO NOT push contents to GitHub
│   ├── raw/                             # Original 7 PCAP CSV files
│   ├── interim/                         # Output of data cleaning
│   └── processed/                       # Final processed dataset
│
├── notebooks/                           
│   └── 01_exploratory_data_analysis.ipynb  # Visualization only
│
└── src/                                 # Pipeline scripts
    ├── __init__.py                      
    ├── config.py                        # Centralized configuration
    ├── 01_data_cleaning.py              
    ├── 02_preprocessing.py              
    └── 03_train_models.py               
```

---

## Team Contributions

- **Tanishi Rai [23BCE10299]**  
  `src/01_data_cleaning.py`  
  Data integration, null/duplicate handling, and memory optimization via downcasting  

- **Kavyansh Vats [23BCE10301]**  
  `src/02_preprocessing.py`  
  Label encoding, feature scaling, and PCA implementation  

- **Ronak Kamboj [23BCE10263]**  
  `src/02_preprocessing.py`, `notebooks/`  
  Discretization (qcut binning) and EDA visualizations  

- **Kasak Singh [23BCE10250]**  
  `src/03_train_models.py`  
  Model training pipeline and dataset splitting  

- **Sanjiban Ghosh [23BCE10267]**  
  `src/03_train_models.py`  
  Model evaluation including classification metrics, ROC curves, and T-tests  

- **DEVASHISH ASWAL [23BCE10273]** - did'nt join the group  

---

## Local Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/DataScience.git
cd DataScience
```

### 2. Install Dependencies
It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Add Raw Data (Required)
Due to size limitations, raw data is not included in the repository.

Place all 7 original `.csv` files into:
```
data/raw/
```

---

## Running the Pipeline

Run the scripts sequentially from the project root directory.

### Step 1: Data Cleaning and Integration
```bash
python src/01_data_cleaning.py
```
- Merges raw CSV files  
- Cleans data  
- Applies memory optimization  
- Outputs: `data/interim/cleaned_data.csv`

---

### Step 2: Preprocessing and Feature Engineering
```bash
python src/02_preprocessing.py
```
- Applies scaling and transformations  
- Performs PCA (95% variance retention)  
- Discretizes features into categories  
- Outputs: `data/processed/final_data.csv`

---

### Step 3: Model Training and Evaluation
```bash
python src/03_train_models.py
```
- Trains machine learning models  
- Generates evaluation metrics  
- Performs statistical testing  
- Outputs ROC curve image (`roc_curve.png`)

---

## Exploratory Data Analysis (EDA)

For visual analysis and insights, use the notebook:

```
notebooks/01_exploratory_data_analysis.ipynb
```

**Note:** Ensure that Steps 1 and 2 have been completed before running the notebook.

---

## Development Guidelines

- Do not modify `config.py` unless the directory structure changes  
- Add new models in `src/03_train_models.py`  
- Add visualizations in the notebook  
- For testing heavy computations, use sampling:
  ```python
  df.sample(n=50000)
  ```
