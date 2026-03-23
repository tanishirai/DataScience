# 📊 Data Science Project – Setup & Contribution Guide

## 🚀 Project Overview

This project includes:

* Data preprocessing (cleaning, transformation)
* Exploratory Data Analysis (EDA)
* Machine Learning models

---

# ⚙️ Getting Started (Run this first)

### 👉 Open Google Colab and run:

```python
!git clone https://github.com/tanishirai/DataScience.git
!git lfs install
%cd /content/DataScience
!git lfs pull
```

---

### 👉 Import libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

# 📂 Load Data (MAIN STEP)

👉 Use the already prepared dataset:

```python
df = pd.read_csv("cleaned_data.csv")
print(df.shape)
df.head()
```

---

# 🧹 About the Dataset

The file `cleaned_data.csv` is already:

* Combined from multiple CSV files
* Cleaned (duplicates, missing values handled)
* Processed (data types fixed, columns cleaned)
* Outliers handled using clipping

👉 So you can directly start analysis or modeling.

---

# ⚠️ Note (Important but simple)

👉 You **do NOT need to:**

* Combine CSV files again
* Run cleaning steps again

👉 The raw + cleaning code is still in the notebook (for reference), but not required to run again.

---

# 👥 For Contributors

## 🟢 How to start

1. Run setup (clone + pull)
2. Load dataset using:

```python
df = pd.read_csv("cleaned_data.csv")
```

---

## 🟡 Start your work below

Continue in the **same notebook**, just add your section like:

```python
# =========================
# EDA SECTION (Your Name)
# =========================
```

or

```python
# =========================
# ML MODEL (Your Name)
# =========================
```

---

## 🟡 Keep things consistent

* Use the same variable → `df`
* Don’t reload or modify dataset unnecessarily
* Don’t repeat cleaning steps
* Keep code simple and readable
* Add small comments if needed

---

## 🟡 Keep notebook organized

* Add clear section headings
* Keep your code in one block (don’t scatter it)
* Follow the same format as others

---


# 💡 Helpful Tips

* If something breaks → restart runtime and run from top once
* Before experimenting:

```python
df_original = df.copy()
```

* If system becomes slow:

```python
df = df.sample(n=200000, random_state=42)
```

---

# 🚀 Workflow Summary

1. Run setup
2. Load cleaned dataset
3. Add your section
4. Continue project flow

---

# 🔥 Final Note

Just keep things simple and consistent.
Everyone works on the same cleaned dataset — that’s the main idea 👍

---
