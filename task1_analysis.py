"""
=======================================================
  EduTech Solutions - AI & ML Internship
  Task 1: Understanding Dataset & Data Types
  Dataset: Students Academic Performance
  Tools: Python (Pandas, NumPy)
=======================================================
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# SECTION 1: Load the Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  TASK 1: UNDERSTANDING DATASET & DATA TYPES")
print("=" * 60)

df = pd.read_csv("students_academic_performance.csv")
print("\n✅ Dataset loaded successfully!")

# ─────────────────────────────────────────────
# SECTION 2: Shape & Dimensions
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📐 SECTION 2: Shape & Dimensions")
print("─" * 60)

print(f"  Rows (Records)   : {df.shape[0]}")
print(f"  Columns (Features): {df.shape[1]}")
print(f"  Total Cells      : {df.size}")
print(f"\n  Column Names:\n  {list(df.columns)}")

# ─────────────────────────────────────────────
# SECTION 3: Data Structure — .info()
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔍 SECTION 3: Data Structure — df.info()")
print("─" * 60)

df.info()

# ─────────────────────────────────────────────
# SECTION 4: Statistical Summary — .describe()
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("📊 SECTION 4: Statistical Summary — df.describe()")
print("─" * 60)

print("\n>>> Numerical Columns:")
print(df.describe().round(2).to_string())

print("\n>>> Categorical Columns (include='object'):")
print(df.describe(include='object').to_string())

# ─────────────────────────────────────────────
# SECTION 5: Data Types Documentation
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("🗂️  SECTION 5: Data Types per Column")
print("─" * 60)

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Dtype": df.dtypes.values,
    "Pandas Type": [str(df[col].dtype) for col in df.columns]
})
print(dtype_df.to_string(index=False))

# ─────────────────────────────────────────────
# SECTION 6: Numerical vs Categorical Variables
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔢 SECTION 6: Numerical vs Categorical Variables")
print("─" * 60)

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n  Numerical Columns ({len(numerical_cols)}):")
for col in numerical_cols:
    is_discrete = df[col].nunique() <= 10
    kind = "Discrete" if is_discrete else "Continuous"
    print(f"    - {col:<22} [{kind}]")

print(f"\n  Categorical Columns ({len(categorical_cols)}):")
for col in categorical_cols:
    print(f"    - {col:<22} [Nominal/Ordinal]")

# ─────────────────────────────────────────────
# SECTION 7: Missing Values Analysis
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("❓ SECTION 7: Missing Values Analysis")
print("─" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
}).sort_values("Missing Count", ascending=False)

missing_df = missing_df[missing_df["Missing Count"] > 0]

if missing_df.empty:
    print("\n  ✅ No missing values found in the dataset!")
else:
    print(f"\n  Total columns with missing values: {len(missing_df)}")
    print(f"  Total missing cells: {missing.sum()}")
    print("\n" + missing_df.to_string())

print("\n  💡 Strategies to handle missing values:")
print("     • Numerical  → Fill with mean/median (df.fillna(df.mean()))")
print("     • Categorical → Fill with mode (df[col].mode()[0])")
print("     • Drop rows  → df.dropna() if < 5% missing")

# ─────────────────────────────────────────────
# SECTION 8: Unique Values in Categorical Columns
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("🏷️  SECTION 8: Unique Values in Categorical Columns")
print("─" * 60)

for col in categorical_cols:
    unique_vals = df[col].dropna().unique()
    print(f"\n  Column: '{col}'")
    print(f"    Unique count : {df[col].nunique()}")
    print(f"    Values       : {list(unique_vals)}")

# ─────────────────────────────────────────────
# SECTION 9: First Few Rows
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("👀 SECTION 9: First 5 Rows of Dataset")
print("─" * 60)
print(df.head().to_string(index=False))

# ─────────────────────────────────────────────
# SECTION 10: Summary Report
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("📋 FINAL SUMMARY REPORT")
print("=" * 60)

summary = f"""
  Dataset       : Students Academic Performance
  Source        : Generated for EduTech AI/ML Internship
  Total Records : {df.shape[0]}
  Total Features: {df.shape[1]}

  DATA TYPES BREAKDOWN:
    • Integer (int64)   : {len(df.select_dtypes('int64').columns)} columns
    • Float (float64)   : {len(df.select_dtypes('float64').columns)} columns
    • Object (string)   : {len(df.select_dtypes('object').columns)} columns

  VARIABLE TYPES:
    • Numerical Columns : {len(numerical_cols)} → {numerical_cols}
    • Categorical Cols  : {len(categorical_cols)} → {categorical_cols}

  MISSING DATA:
    • Columns affected  : {len(missing_df)} columns
    • Total missing     : {missing.sum()} cells
    • Missing %         : {(missing.sum() / df.size * 100):.2f}% of all data

  KEY OBSERVATIONS:
    1. CGPA and AttendancePercent are continuous numerical variables.
    2. MathScore, ScienceScore, EnglishScore are discrete numerical (integer scores).
    3. Gender, Department, Grade, City are nominal categorical variables.
    4. Year is discrete — could be encoded as ordinal (1 < 2 < 3 < 4).
    5. ScholarshipHolder and PartTimeJob are binary categorical (Yes/No).
    6. Missing values are present in AttendancePercent, MathScore, CGPA, and City.
"""
print(summary)

print("=" * 60)
print("✅ Task 1 Complete! All steps executed successfully.")
print("=" * 60)
