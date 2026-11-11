# %% [markdown]
# # Week 2 - Mini Semantic Search over FAQs
#
# **Pipeline:** load FAQs with Pandas -> clean -> chunk the text -> embed + store in Chroma -> query -> print the top 3 matches.
#
# *Semantic* search finds answers by **meaning**, not by exact words: the query "I lost my password"
# should find the FAQ "I forgot my student portal password" even though the words are different.
#
# Run as a script: `python semantic_search.py`  |  Run in Jupyter/Colab: run the cells top to bottom.
# The first run downloads a small embedding model (about 80 MB), so it needs internet.

# %% [markdown]
# ## 0. Setup

# %%
#NB# %pip install -q chromadb pandas

# %%
import html
import re
from pathlib import Path

import chromadb
import pandas as pd

# Works both as a script (uses __file__) and inside a notebook (uses the current folder)
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
FAQ_PATH = BASE_DIR / "data" / "faqs_raw.csv"

#NB# # In Colab the repo files are not there by default: upload faqs_raw.csv when asked.
#NB# if not FAQ_PATH.exists():
#NB#     from google.colab import files
#NB#     print("Please upload faqs_raw.csv")
#NB#     uploaded = files.upload()
#NB#     FAQ_PATH = BASE_DIR / next(iter(uploaded))

CHUNK_WORDS = 45      # maximum words in one chunk
CHUNK_OVERLAP = 10    # words shared between neighbouring chunks (keeps context at the cut)
TOP_K = 3             # how many matches to show
COLLECTION_NAME = "campus_faqs"

# %% [markdown]
# ## 1. Load the FAQ dataset with Pandas

# %%
def load_faqs(path):
    """Read the raw FAQ file (columns: faq_id, category, question, answer)."""
    return pd.read_csv(path)


raw_faqs = load_faqs(FAQ_PATH)
print(f"Loaded {len(raw_faqs)} FAQ rows")
print(raw_faqs[["faq_id", "category", "question"]].head(8).to_string(index=False))

# %% [markdown]
# ## 2. Clean the text
# The raw file has HTML tags, extra spaces, inconsistent category names, a duplicate question and an empty answer.

# %%
CATEGORY_FIX = {"it services": "IT Services"}


def clean_text(value):
    """Remove HTML tags, decode HTML symbols, collapse whitespace."""
    value = html.unescape(str(value))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def clean_faqs(df):
    """Return a clean copy: tidy text, same category spelling, no empty answers, no duplicates."""
    df = df.copy()
    df["question"] = df["question"].fillna("").map(clean_text)
    df["answer"] = df["answer"].fillna("").map(clean_text)

    key = df["category"].fillna("").str.strip().str.lower()
    df["category"] = key.map(CATEGORY_FIX).fillna(key.str.title())

    before = len(df)
    df = df[(df["question"] != "") & (df["answer"] != "")]
    print(f"[clean] dropped {before - len(df)} row(s) with an empty question or answer")

    before = len(df)
    df = df.assign(_q=df["question"].str.lower()).drop_duplicates(subset="_q", keep="first").drop(columns="_q")
    print(f"[clean] removed {before - len(df)} duplicate question(s)")
    return df.reset_index(drop=True)


faqs = clean_faqs(raw_faqs)
print(f"{len(raw_faqs)} raw rows -> {len(faqs)} clean FAQs")
print("Categories:", sorted(faqs["category"].unique()))

# %% [markdown]
# ## 3. Chunk the text
# Embedding models work best on short passages, so long answers are cut into overlapping chunks of about 45 words.
# Each chunk also gets its question in front of it, so a chunk never loses the topic.

# %%
def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Split text into chunks of `chunk_words` words; neighbours share `overlap` words."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks


def make_chunk_records(df):
    """Create ids, documents and metadata lists in the format Chroma expects."""
    ids, documents, metadatas = [], [], []
    for row in df.itertuples():
        for i, chunk in enumerate(chunk_text(row.answer)):
            ids.append(f"faq{row.faq_id}-chunk{i}")
            documents.append(f"{row.question}\n{chunk}")
            metadatas.append({
                "faq_id": int(row.faq_id),
                "category": row.category,
                "question": row.question,
                "chunk_index": i,
            })
    return ids, documents, metadatas


ids, documents, metadatas = make_chunk_records(faqs)
print(f"{len(faqs)} FAQs -> {len(documents)} chunks")
print()
print("Example chunk:")
print(documents[0])

# %% [markdown]
# ## 4. Embed and store the chunks in Chroma
# Chroma turns every chunk into a vector (an *embedding*) with its default model and stores it.
# Chunks with a similar meaning end up close to each other.

# %%
def build_collection(ids, documents, metadatas):
    """Create an in-memory Chroma collection and add all chunks (embedding happens automatically)."""
    client = chromadb.Client()  # in memory; use chromadb.PersistentClient(path="chroma_db") to keep it on disk
    try:
        client.delete_collection(COLLECTION_NAME)  # start fresh if the cell is run twice
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


collection = build_collection(ids, documents, metadatas)
print(f"Stored {collection.count()} chunks in Chroma collection '{COLLECTION_NAME}'")

# %% [markdown]
# ## 5. Query it and print the top 3 matches
# `distance` is how far the chunk is from the query: **lower = more similar**.
# One FAQ can have several chunks, so we keep only the best chunk per FAQ.

# %%
def search_faqs(collection, query, top_k=TOP_K, category=None):
    """Return the top_k best-matching FAQs (one per FAQ) for a query."""
    kwargs = {"query_texts": [query], "n_results": min(top_k * 3, collection.count())}
    if category:
        kwargs["where"] = {"category": category}  # optional filter on metadata
    result = collection.query(**kwargs)

    matches, seen = [], set()
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        if meta["faq_id"] in seen:
            continue
        seen.add(meta["faq_id"])
        matches.append({"faq_id": meta["faq_id"], "category": meta["category"],
                        "question": meta["question"], "text": doc.split("\n", 1)[1], "distance": dist})
        if len(matches) == top_k:
            break
    return matches


def print_matches(query, matches):
    print(f'Query: "{query}"')
    for rank, m in enumerate(matches, start=1):
        snippet = m["text"][:130] + ("..." if len(m["text"]) > 130 else "")
        print(f"  {rank}. [{m['category']}] {m['question']}  (distance {m['distance']:.3f})")
        print(f"     {snippet}")
    print()


example_queries = [
    "How do I get on the wifi?",
    "I lost my password",
    "Can I get money to pay for my studies?",
    "Where can I practice coding with other students?",
    "I feel anxious about my exams",
]
for q in example_queries:
    print_matches(q, search_faqs(collection, q))

# %% [markdown]
# ### Extra: filter by category and try your own query

# %%
print_matches("How many days do I have?", search_faqs(collection, "How many days do I have?", category="Exams"))

my_query = "my phone is not allowed?"  # <- change this and run the cell again
print_matches(my_query, search_faqs(collection, my_query))

# %% [markdown]
# ### Extra: quick quality check (hit rate)
# For a few queries we know which FAQ *should* come first. A "hit" means the right FAQ is in the top 3.

# %%
expected = {
    "How do I get on the wifi?": 7,
    "I lost my password": 8,
    "Can I get money to pay for my studies?": 16,
    "Where can I practice coding with other students?": 19,
    "I feel anxious about my exams": 21,
    "Where do I get a transcript?": 25,
}
hits = 0
for q, faq_id in expected.items():
    top_ids = [m["faq_id"] for m in search_faqs(collection, q)]
    hit = faq_id in top_ids
    hits += hit
    print(f"{'HIT ' if hit else 'MISS'}  {q!r} -> expected FAQ {faq_id}, got {top_ids}")
print(f"\nTop-3 hit rate: {hits}/{len(expected)}")

# %% [markdown]
# ## What I learned
# - Cleaning first matters: HTML tags and duplicate questions would add noise to the embeddings.
# - Chunking with overlap keeps long answers searchable without losing the sentence at the cut.
# - Semantic search matches meaning ("lost my password" finds "forgot my password") where keyword search would fail.
# - A tiny hit-rate check is an easy way to notice when a change makes the search worse.
