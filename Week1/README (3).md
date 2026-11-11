# Week 1 - Small Python Data-Processing App

A small app that **loads a raw dataset, cleans it with functions, and processes it with NumPy**.
It runs as a plain Python script and as a Jupyter/Colab notebook.

The dataset (`data/students_raw.csv`) is **synthetic** student data that I made messy on purpose:
inconsistent text/case, duplicates, missing values, text inside number columns, mixed date formats and impossible values.

## Files

| File | What it is |
|------|------------|
| `data_processing_app.py` | The app as a Python script |
| `data_processing_app.ipynb` | Same code as a Colab/Jupyter notebook, with saved outputs |
| `data/students_raw.csv` | Raw messy dataset (64 rows) |
| `data/students_clean.csv` | Output of the app (59 clean rows) |

## How to run

### Locally (with a virtual environment)

```bash
# from the repository root
python -m venv venv

# activate it
#   Windows (PowerShell):  venv\Scripts\Activate.ps1
#   Windows (cmd):         venv\Scripts\activate.bat
#   macOS / Linux:         source venv/bin/activate

pip install -r requirements.txt
cd week-01
python data_processing_app.py
```

The `venv/` folder is listed in `.gitignore`, so it is never committed.

### In Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com) -> **File -> Upload notebook** -> choose `data_processing_app.ipynb`
   (or use the **GitHub** tab and paste the repository URL).
2. Run all cells (**Runtime -> Run all**). When asked, upload `students_raw.csv`.

## What the app does

**Cleaning functions** (each does one job):

| Function | Problem it fixes |
|----------|------------------|
| `fix_text_columns` | extra spaces, UPPER/lower case, 5 spellings of one department, blanks -> NaN |
| `drop_unidentifiable_rows` | rows with no student ID or name |
| `remove_duplicates` | the same student listed twice (even with different capitalisation) |
| `convert_types` | `"84.5%"` -> `84.5`, `"N/A"` -> missing, text dates -> real dates |
| `remove_outliers` | impossible values (age 150, score 999) and extreme IQR outliers |
| `fill_missing_values` | fills missing numbers with the column median |

**NumPy processing functions:** `compute_statistics` (mean, median, std, min, max), `compute_percentiles`,
`add_average_score`, `assign_letter_grades` (`np.select`), `attendance_score_correlation` (`np.corrcoef`),
`top_students` (`np.argsort`), `find_at_risk_students` (z-scores + `np.where`), `department_summary` (boolean masks).

## Results from my run

- 64 raw rows -> **59 clean rows** (1 row without a name dropped, 4 duplicates removed).
- 6 outlier values replaced, then all missing numbers filled (the one missing enrolment date is kept as empty on purpose).
- Correlation between attendance and average score: **0.42** (students who attend more tend to score higher).

## What I learned

- Real data is messy, and the order of cleaning steps matters (fix text first, then remove duplicates).
- Small single-purpose functions are easy to test and reuse.
- The median is a safer fill value than the mean when outliers exist.
