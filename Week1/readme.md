# AWS-sbg-week-1-task

Small Python Data-Processing App.

This repository contains a beginner-friendly week 1 assignment for loading, cleaning, and analyzing a small student dataset using Python and NumPy.

## Project overview

The goal of this project is to practice real-world data-cleaning steps in a simple, readable way:

- load a CSV file
- clean inconsistent text values
- convert text values to the correct types
- handle missing values carefully
- remove duplicate rows
- validate numerical data and fix invalid entries
- calculate summary statistics with NumPy
- present results in a Jupyter notebook

## Repository structure

```text
AWS-sbg-week-1-task/
├── README.md
├── requirements.txt
├── .gitignore
└── week-01/
    ├── README.md
    ├── task.py
    ├── task.ipynb
    └── students.csv
```

## Technologies used

- Python 3
- CSV module from the Python standard library
- NumPy for numerical processing
- Jupyter Notebook / Google Colab

## Setup instructions

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Python program

From the repository root:

```bash
python week-01/task.py
```

This script loads the raw student dataset, cleans it, removes duplicates, validates values, and prints summary statistics.

## Open and run the notebook

You can open the notebook in Jupyter or Google Colab.

### Jupyter

```bash
jupyter notebook week-01/task.ipynb
```

### Google Colab

1. Upload the notebook file to Google Colab.
2. If `students.csv` is not already in the notebook folder, upload it as well.
3. Run the cells from top to bottom.
4. Save the notebook to keep the outputs.

The notebook includes:

- project explanation
- raw data preview
- cleaning steps
- before/after comparisons
- NumPy statistics
- a short conclusion

## Dataset description

The dataset is a small CSV file with this structure:

```text
name,age,score
```

The dataset intentionally includes:

- missing values
- duplicate rows
- inconsistent spacing and capitalization
- numeric values stored as strings
- invalid age or score values
- non-numeric entries in numeric columns

These problems are included so the cleaning process is easy to understand and explain.

## Cleaning methods

The project uses a simple, beginner-friendly cleaning workflow:

1. Load CSV rows with `csv.DictReader`.
2. Standardize names by trimming spaces and capitalizing correctly.
3. Convert text values such as `"18"` to numbers.
4. Replace missing values with a median when appropriate.
5. Remove duplicate rows using a normalized key.
6. Validate age and score ranges.
7. Use NumPy to compute statistics.

A score is considered valid only within a reasonable range for the dataset, such as 0 to 100. Age values are also checked for realistic ranges before imputation or correction.

## Example output

The script prints a result similar to this when it runs successfully:

```text
Raw records loaded: 15
Cleaned records: 12
Mean score: 72.50
Median score: 74.00
Standard deviation: 12.88
90th percentile: 88.40
Min score: 45.00
Max score: 95.00
```

## Notes

- The project does not commit a virtual environment.
- Environment files are ignored with `.gitignore`.
- The code is intentionally clear and beginner-friendly so it can be explained confidently to a mentor.
