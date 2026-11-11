# Week 2 - Mini Semantic Search

A small semantic search engine over a campus-services FAQ. You type a question in your own words and it returns
the **3 most relevant FAQs by meaning**, not by exact keywords.

**Pipeline:** load FAQs (Pandas) -> clean -> chunk -> embed and store in Chroma -> query -> print the top 3 matches.

The FAQ file (`data/faqs_raw.csv`) is **sample data I wrote** for practice. It is deliberately messy
(HTML tags, extra spaces, inconsistent category names, a duplicate question, an empty answer).

## Files

| File | What it is |
|------|------------|
| `semantic_search.py` | The app as a Python script |
| `semantic_search.ipynb` | Same code as a Colab/Jupyter notebook |
| `data/faqs_raw.csv` | Raw FAQ dataset (28 rows) |

## How to run

### Locally (virtual environment)

```bash
# from the repository root
python -m venv venv
#   Windows (PowerShell): venv\Scripts\Activate.ps1   |   macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
cd week-02
python semantic_search.py
```

The first run downloads Chroma's small default embedding model (about 80 MB), so you need internet once.

### In Google Colab

1. **File -> Upload notebook** -> `semantic_search.ipynb`.
2. **Runtime -> Run all**. The first cell installs `chromadb`; when asked, upload `faqs_raw.csv`.
3. Download the executed notebook (**File -> Download -> Download .ipynb**) and commit it so the outputs are saved.

## How it works

| Step | Function | What happens |
|------|----------|--------------|
| 1 | `load_faqs` | reads the CSV with Pandas |
| 2 | `clean_faqs` / `clean_text` | removes HTML, fixes spaces and category spelling, drops the empty answer and the duplicate question (28 -> 25 FAQs) |
| 3 | `chunk_text` / `make_chunk_records` | cuts long answers into overlapping 45-word chunks, each starting with its question (25 FAQs -> 33 chunks) |
| 4 | `build_collection` | Chroma embeds every chunk and stores it |
| 5 | `search_faqs` / `print_matches` | embeds the query, finds the nearest chunks, keeps the best chunk per FAQ, prints the top 3 |

Extras included: filtering by category (`where={"category": ...}`) and a small hit-rate check.

## Example queries

Run these in the notebook; the expected best match is shown in the last column.
`distance` in the output means **lower = more similar**.

| Query | Expected FAQ |
|-------|--------------|
| `How do I get on the wifi?` | How do I connect to the campus Wi-Fi? |
| `I lost my password` | I forgot my student portal password. What should I do? |
| `Can I get money to pay for my studies?` | Are scholarships available and how do I apply? |
| `Where can I practice coding with other students?` | Where can I find information about hackathons and coding competitions? |
| `I feel anxious about my exams` | Who can I talk to if I feel stressed or overwhelmed? |
| `Where do I get a transcript?` | How do I get an official transcript? |

Notice that none of these queries repeats the FAQ wording, which is the point of semantic search.
The last cell of the notebook prints the real **top-3 hit rate** for these queries.

## What I learned

- Cleaning before embedding matters: HTML tags and duplicate questions add noise.
- Overlapping chunks keep long answers searchable without losing the sentence at the cut.
- Embeddings match meaning, so "lost my password" finds "forgot my password" where keyword search fails.
