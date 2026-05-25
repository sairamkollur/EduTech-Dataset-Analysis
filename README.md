# 📊 EduTech Solutions — AI & ML Internship
## Task 1: Understanding Dataset & Data Types

---

## 🎯 Objective
Build a strong foundation in Exploratory Data Analysis (EDA) by understanding the structure, types, and quality of a dataset using Python (Pandas & NumPy).

---

## 📁 Repository Structure

```
task1-dataset-analysis/
│
├── Task1_EduTech_Dataset_Analysis.ipynb   # Main Jupyter Notebook
├── task1_analysis.py                      # Python Script version
├── students_academic_performance.csv      # Dataset used
└── README.md                              # This file
```

---

## 📦 Dataset: Students Academic Performance

A custom dataset with **200 student records** and **15 features** covering academic and demographic information.

| Column | Type | Description |
|--------|------|-------------|
| StudentID | int64 | Unique student identifier |
| Name | object | Student name |
| Age | int64 | Student age (17–24) |
| Gender | object | Male / Female / Other |
| Department | object | CS / Math / Physics / Chemistry / Biology |
| Year | int64 | Academic year (1–4) |
| AttendancePercent | float64 | Attendance percentage |
| MathScore | float64 | Math exam score |
| ScienceScore | int64 | Science exam score |
| EnglishScore | int64 | English exam score |
| CGPA | float64 | Cumulative GPA (4.0–10.0) |
| ScholarshipHolder | object | Yes / No |
| PartTimeJob | object | Yes / No |
| Grade | object | A / B / C / D / F |
| City | object | Student's home city |

---

## 🔍 What This Task Covers

### Step 1 — Import Libraries
```python
import pandas as pd
import numpy as np
```

### Step 2 — Load Dataset
```python
df = pd.read_csv("students_academic_performance.csv")
df.head()
```

### Step 3 — Shape & Dimensions
```python
df.shape   # (200, 15)
df.size    # 3000
```

### Step 4 — Data Structure
```python
df.info()  # Shows dtypes, non-null counts, memory usage
```

### Step 5 — Data Types
- `int64` → StudentID, Age, Year, ScienceScore, EnglishScore
- `float64` → AttendancePercent, MathScore, CGPA
- `object` → Name, Gender, Department, Grade, City, etc.

### Step 6 — Statistical Summary
```python
df.describe()              # numerical columns
df.describe(include='object')  # categorical columns
```

### Step 7 — Numerical vs Categorical
- **Continuous**: CGPA, AttendancePercent
- **Discrete**: Age, Year, Scores
- **Nominal**: Gender, City, Department
- **Ordinal**: Grade (A > B > C > D > F)

### Step 8 — Missing Values
```python
df.isnull().sum()
```
| Column | Missing | % |
|--------|---------|---|
| AttendancePercent | 15 | 7.5% |
| MathScore | 10 | 5.0% |
| CGPA | 8 | 4.0% |
| City | 5 | 2.5% |

### Step 9 — Unique Values in Categorical Columns
```python
for col in categorical_cols:
    print(df[col].value_counts())
```

---

## 📋 Key Findings

1. Dataset has **200 rows** and **15 columns**
2. **5 integer**, **3 float**, and **7 object** columns
3. **4 columns** have missing values (38 total, 1.27% of data)
4. `Grade` is **ordinal** categorical (A > B > C > D > F)
5. `ScholarshipHolder` and `PartTimeJob` are **binary** categorical
6. `Year` (1–4) is discrete but could be encoded as ordinal

---

## 🎙️ Interview Q&A

**Q: What is the difference between structured and unstructured data?**  
Structured data is organized in rows and columns (our CSV). Unstructured data has no fixed format — like text, images, audio.

**Q: What are continuous and discrete variables?**  
Continuous: infinite values in a range (CGPA = 7.23, 8.501...). Discrete: countable whole values (Year = 1, 2, 3, 4).

**Q: How do you handle missing values?**  
Numerical → fill with mean/median. Categorical → fill with mode. If < 5% missing → can drop rows.

**Q: What does .describe() tell you?**  
Provides count, mean, standard deviation, min, max, and quartiles for numerical columns. Helps understand data spread and detect outliers.

---

## 🛠️ How to Run

```bash
# Clone the repo
git clone https://github.com/sairamkollur/EduTech-Dataset-Analysis.git

# Install dependencies
pip install pandas numpy jupyter

# Open notebook
jupyter notebook Task1_EduTech_Dataset_Analysis.ipynb

# OR run the Python script
python task1_analysis.py
```

---

## 👤 Author
**Kollur Sai Ram**  
EduTech Solutions — AI & ML Internship, 2026
