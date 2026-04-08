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

## 👥 Team Contributions

This project was built collaboratively, covering an end-to-end data science lifecycle. Below are the specific contributions and model ownership for each team member:

* **Tanishi Rai [23BCE10299]** *Files:* `src/01_data_cleaning.py`, `src/03_train_models.py`  
  * *Data Engineering:* Handled data integration, null/duplicate handling, and memory optimization via aggressive downcasting.  
  * *Machine Learning:* Implemented the **Naive Bayes Classifier** (GaussianNB) leveraging the scaled PCA features.

* **Kavyansh Vats [23BCE10301]** *Files:* `src/02_preprocessing.py`, `src/03_train_models.py`  
  * *Data Transformation:* Engineered the Label Encoding, Feature Scaling (StandardScaler), and Principal Component Analysis (PCA).  
  * *Machine Learning:* Developed and trained the **Decision Tree Classifier** for non-linear predictions.

* **Ronak Kamboj [23BCE10263]** *Files:* `src/02_preprocessing.py`, `src/03_train_models.py`, `notebooks/`  
  * *Data Discretization & EDA:* Handled `qcut` binning and built the Exploratory Data Analysis visualizations.  
  * *Model Visualization:* Extracted prediction arrays to generate comparative **Confusion Matrices** and **K-Means Scatter Plots**.

* **Kasak Singh [23BCE10250]** *Files:* `src/03_train_models.py`  
  * *Pipeline Setup:* Architected the core model training pipeline and 80/20 dataset splitting logic.  
  * *Machine Learning:* Tuned and trained both the **Logistic Regression** baseline classifier and the **K-Nearest Neighbors (KNN)** model.

* **Sanjiban Ghosh [23BCE10267]** *Files:* `src/03_train_models.py`  
  * *Unsupervised Learning:* Engineered the **K-Means Clustering** model to map natural groupings in the network traffic.  
  * *Statistical Evaluation:* Built the rigorous evaluation suite, including classification metrics, multiclass **ROC Curves**, and Paired **T-Tests** for model comparison.

- **Devashish Aswal [23BCE10273]** - didn't join the group  

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

**Note:** Ensure that Steps 1 and 2 have been completed before running the jupyter notebook.

---

## Development Guidelines

- Do not modify `config.py` unless the directory structure changes  
- Add new models in `src/03_train_models.py`  
- Add visualizations in the notebook  
- For testing heavy computations, use sampling:
  ```python
  df.sample(n=50000)
  ```
