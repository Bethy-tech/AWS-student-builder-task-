# AWS-student-builder-task-
# aws-sbg-tasks

A collection of my weekly tasks and projects for the AWS SBG program.
Each week has its own folder with the Python source code, a Google Colab notebook, and a short README that explains how to run it.

## Repository structure

```
aws-sbg-tasks/
├── README.md
├── requirements.txt
├── .gitignore
├── week-01/
│   ├── README.md
│   ├── <task_name>.py
│   └── <task_name>.ipynb
├── week-02/
│   └── ...
└── ...
```

- `week-XX/` holds everything for that week's task.
- The `.py` file and the `.ipynb` notebook contain the same code.
- Each week's `README.md` explains what the task does, how to run it, and what I learned.

## Getting started

```bash
# clone the repository
git clone https://github.com/<your-username>/aws-sbg-tasks.git
cd aws-sbg-tasks

# create and activate a virtual environment
python -m venv venv
#   Windows (PowerShell):  venv\Scripts\Activate.ps1
#   macOS / Linux:         source venv/bin/activate

# install the dependencies
pip install -r requirements.txt
```

Then open a week's folder and follow its README:

```bash
cd week-01
python <task_name>.py
```

## Running the notebooks in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Choose **File -> Upload notebook** and select the `.ipynb` file (or use the **GitHub** tab and paste this repository's URL).
3. Run all cells with **Runtime -> Run all**.

## Conventions

- One folder per week, named `week-01`, `week-02`, and so on.
- Every week includes both a `.py` file and a Colab `.ipynb` notebook that runs and shows its outputs.
- Files have clear names, and the code is commented and runs from top to bottom.
- Commits are small and frequent, so the Git history shows real progress.
- Secrets such as API keys are never committed; they are read from environment variables or a git-ignored `.env` file.
- Large or public datasets are linked in the README, not uploaded.
- Virtual environments and cache files (`venv/`, `__pycache__/`, `.ipynb_checkpoints/`) are listed in `.gitignore`.

## Tools used

Python, Jupyter / Google Colab, Git and GitHub. The Python packages are listed in `requirements.txt`.

## Note

AI tools may be used to help with learning and coding, but I make sure I understand and can explain everything I submit.
