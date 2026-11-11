# %% [markdown]
# # Week 1 - Small Python Data-Processing App
#
# **What this app does**
# 1. Loads a raw, messy dataset (`data/students_raw.csv`, synthetic student records).
# 2. Cleans it with small named functions (text/case, duplicates, types, outliers, missing values).
# 3. Processes the clean data with NumPy (mean, median, std, percentiles, correlation, grades...).
# 4. Saves the clean file to `data/students_clean.csv`.
#
# Run as a script: `python data_processing_app.py`  |  Run in Jupyter/Colab: run the cells top to bottom.

# %% [markdown]
# ## 0. Setup

# %%
#NB# %pip install -q numpy pandas

# %%
from pathlib import Path

import numpy as np
import pandas as pd

# Works both as a script (uses __file__) and inside a notebook (uses the current folder)
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RAW_PATH = BASE_DIR / "data" / "students_raw.csv"
CLEAN_PATH = BASE_DIR / "data" / "students_clean.csv"

#NB# # In Colab the repo files are not there by default: upload students_raw.csv when asked.
#NB# if not RAW_PATH.exists():
#NB#     from google.colab import files
#NB#     print("Please upload students_raw.csv")
#NB#     uploaded = files.upload()
#NB#     RAW_PATH = BASE_DIR / next(iter(uploaded))

SCORE_COLS = ["math_score", "programming_score", "english_score"]
NUMERIC_COLS = ["age", "attendance_percent"] + SCORE_COLS

# Hard limits: anything outside these ranges is impossible (e.g. age 150, score 999)
VALID_RANGES = {
    "age": (15, 60),
    "attendance_percent": (0, 100),
    "math_score": (0, 100),
    "programming_score": (0, 100),
    "english_score": (0, 100),
}

# Different spellings of the same department -> one standard name
DEPARTMENT_MAP = {
    "software engineering": "Software Engineering",
    "software eng.": "Software Engineering",
    "soft. eng": "Software Engineering",
    "computer science": "Computer Science",
    "comp sci": "Computer Science",
    "cs": "Computer Science",
    "information systems": "Information Systems",
    "info systems": "Information Systems",
    "electrical engineering": "Electrical Engineering",
    "electrical eng.": "Electrical Engineering",
    "ee": "Electrical Engineering",
}

# %% [markdown]
# ## 1. Load the raw dataset
# Everything is read as text first (`dtype=str`) so pandas does not guess types wrongly.
# We convert the types ourselves later, in a function.

# %%
def load_raw_data(path):
    """Read the raw CSV as text. Empty cells stay as empty strings for now."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


raw_df = load_raw_data(RAW_PATH)
print(f"Loaded {len(raw_df)} rows and {raw_df.shape[1]} columns from {RAW_PATH.name}")
print()
print(raw_df.head(8).to_string())

# %% [markdown]
# ## 2. Clean the data with functions
# Each function does **one** job and returns a new DataFrame, so we can test them one by one.
#
# Order matters: fix text first -> remove duplicates -> convert types -> remove outliers -> fill missing values.

# %%
def fix_text_columns(df):
    """Trim spaces, fix upper/lower case, standardise department names, turn blanks into NaN."""
    df = df.copy()
    for col in ["student_id", "name", "department", "enrolled_on"]:
        df[col] = df[col].str.strip()
    df["student_id"] = df["student_id"].str.upper()
    df["name"] = df["name"].str.replace(r"\s+", " ", regex=True).str.title()
    dept_key = df["department"].str.lower()
    df["department"] = dept_key.map(DEPARTMENT_MAP).fillna(df["department"].str.title())
    df = df.replace("", np.nan)  # blank cells are missing values
    print("[clean] text fixed: spaces trimmed, names/IDs/departments standardised")
    return df


def drop_unidentifiable_rows(df):
    """A row without a student ID or a name cannot be used, so we drop it."""
    before = len(df)
    df = df.dropna(subset=["student_id", "name"])
    print(f"[clean] dropped {before - len(df)} row(s) with no student ID or name")
    return df


def remove_duplicates(df):
    """Keep the first row for each student ID (works after IDs are upper-cased)."""
    before = len(df)
    df = df.drop_duplicates(subset="student_id", keep="first")
    print(f"[clean] removed {before - len(df)} duplicate row(s)")
    return df


def convert_types(df):
    """Turn text into real numbers and dates. Things like 'N/A' become NaN (missing)."""
    df = df.copy()
    df["attendance_percent"] = df["attendance_percent"].str.replace("%", "", regex=False)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["enrolled_on"] = pd.to_datetime(df["enrolled_on"], errors="coerce", format="mixed")
    print("[clean] types converted: numbers -> float, enrolled_on -> datetime")
    return df


def outlier_bounds(values, k=3.0):
    """IQR rule with NumPy: values further than k*IQR from the quartiles are extreme outliers."""
    q1, q3 = np.nanpercentile(values, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def remove_outliers(df, columns):
    """Replace impossible values (outside VALID_RANGES) and extreme outliers (IQR) with NaN."""
    df = df.copy()
    total = 0
    for col in columns:
        low, high = VALID_RANGES[col]
        impossible = (df[col] < low) | (df[col] > high)
        df.loc[impossible, col] = np.nan

        iqr_low, iqr_high = outlier_bounds(df[col].to_numpy(dtype=float))
        extreme = (df[col] < iqr_low) | (df[col] > iqr_high)
        df.loc[extreme, col] = np.nan
        total += int(impossible.sum() + extreme.sum())
    print(f"[clean] {total} outlier value(s) replaced with NaN (filled in the next step)")
    return df


def missing_report(df):
    """Count missing values per column (only columns that have some)."""
    counts = df.isna().sum()
    return counts[counts > 0]


def fill_missing_values(df):
    """Fill missing numbers with the column median (robust against outliers)."""
    df = df.copy()
    for col in NUMERIC_COLS:
        median = np.nanmedian(df[col].to_numpy(dtype=float))
        df[col] = df[col].fillna(round(float(median), 1))
    print("[clean] missing numeric values filled with the column median")
    return df


def clean_data(df):
    """Run every cleaning step in the right order."""
    df = fix_text_columns(df)
    df = drop_unidentifiable_rows(df)
    df = remove_duplicates(df)
    df = convert_types(df)
    print()
    print("Missing values BEFORE outlier removal / filling:")
    print(missing_report(df).to_string())
    print()
    df = remove_outliers(df, NUMERIC_COLS)
    df = fill_missing_values(df)
    return df.reset_index(drop=True)


clean_df = clean_data(raw_df)
print()
print(f"Rows: {len(raw_df)} raw -> {len(clean_df)} clean")
print("Missing numeric values left:", int(clean_df[NUMERIC_COLS].isna().sum().sum()))
print("Missing enrolment dates left (kept as NaT on purpose):", int(clean_df["enrolled_on"].isna().sum()))

# %% [markdown]
# ### Look at the cleaned data

# %%
print(clean_df.dtypes.to_string())
print()
print(clean_df.head(8).to_string())
print()
print("Departments now:", sorted(clean_df["department"].unique()))

# %% [markdown]
# ## 3. Process the clean data with NumPy
# We move the numbers into NumPy arrays and calculate with NumPy functions
# (`np.mean`, `np.median`, `np.std`, `np.percentile`, `np.select`, `np.corrcoef`, `np.argsort`, `np.where`).

# %%
def compute_statistics(df, columns):
    """Mean, median, standard deviation, min and max for each column."""
    rows = {}
    for col in columns:
        values = df[col].to_numpy(dtype=float)
        rows[col] = {
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }
    return pd.DataFrame(rows).T.round(2)


def compute_percentiles(df, columns, percentiles=(25, 50, 75, 90)):
    """Percentiles tell us e.g. the score that 90% of students are below."""
    rows = {}
    for col in columns:
        values = df[col].to_numpy(dtype=float)
        rows[col] = dict(zip([f"p{p}" for p in percentiles], np.percentile(values, percentiles)))
    return pd.DataFrame(rows).T.round(2)


def add_average_score(df):
    """Average of the three subject scores per student (mean across each row)."""
    df = df.copy()
    df["average_score"] = np.mean(df[SCORE_COLS].to_numpy(dtype=float), axis=1).round(1)
    return df


def assign_letter_grades(averages):
    """Turn averages into letter grades using np.select (like a vectorised if/elif)."""
    conditions = [averages >= 85, averages >= 75, averages >= 65, averages >= 50]
    return np.select(conditions, ["A", "B", "C", "D"], default="F")


def attendance_score_correlation(df):
    """Pearson correlation between attendance and average score (-1 to +1)."""
    attendance = df["attendance_percent"].to_numpy(dtype=float)
    average = df["average_score"].to_numpy(dtype=float)
    return float(np.corrcoef(attendance, average)[0, 1])


def top_students(df, n=5):
    """Indexes of the n highest averages using np.argsort."""
    order = np.argsort(-df["average_score"].to_numpy(dtype=float))[:n]
    return df.iloc[order][["student_id", "name", "department", "average_score"]]


def find_at_risk_students(df, z_limit=-1.0):
    """Students whose average is more than 1 standard deviation below the mean (z-score)."""
    average = df["average_score"].to_numpy(dtype=float)
    z_scores = (average - np.mean(average)) / np.std(average)
    at_risk = np.where(z_scores < z_limit)[0]  # positions where the condition is True
    return df.iloc[at_risk][["student_id", "name", "average_score"]].assign(z_score=z_scores[at_risk].round(2))


def department_summary(df):
    """Average score and student count per department using boolean masks."""
    departments = np.unique(df["department"].to_numpy())
    rows = []
    for dept in departments:
        mask = (df["department"] == dept).to_numpy()
        rows.append({
            "department": dept,
            "students": int(mask.sum()),
            "avg_score": round(float(np.mean(df.loc[mask, "average_score"].to_numpy(dtype=float))), 1),
            "avg_attendance": round(float(np.mean(df.loc[mask, "attendance_percent"].to_numpy(dtype=float))), 1),
        })
    return pd.DataFrame(rows)

# %%
final_df = add_average_score(clean_df)
final_df["grade"] = assign_letter_grades(final_df["average_score"].to_numpy(dtype=float))

print("=== Statistics (NumPy) ===")
print(compute_statistics(final_df, NUMERIC_COLS + ["average_score"]).to_string())
print()
print("=== Percentiles ===")
print(compute_percentiles(final_df, SCORE_COLS + ["average_score"]).to_string())
print()
print("=== Grade distribution ===")
print(final_df["grade"].value_counts().sort_index().to_string())
print()
print(f"Correlation between attendance and average score: {attendance_score_correlation(final_df):.2f}")
print()
print("=== Top 5 students ===")
print(top_students(final_df, 5).to_string(index=False))
print()
print("=== Students more than 1 std below the mean (at risk) ===")
print(find_at_risk_students(final_df).to_string(index=False))
print()
print("=== Department summary ===")
print(department_summary(final_df).to_string(index=False))

# %% [markdown]
# ## 4. Save the cleaned data

# %%
CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(CLEAN_PATH, index=False)
print(f"Saved {len(final_df)} clean rows to {CLEAN_PATH.name}")

# %% [markdown]
# ## 5. What I learned
# - Real data is messy: the same department was written 4-5 different ways, and some values were impossible (age 150, score 999).
# - Small functions that do one job are easy to test and reuse; `clean_data` just calls them in order.
# - The median is a safer fill value than the mean when outliers exist.
# - NumPy does the maths on whole arrays at once (no slow Python loops), e.g. `np.corrcoef` and `np.select`.
