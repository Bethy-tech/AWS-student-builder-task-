# %% [markdown]
# # Week 3 - Break the Spam Filter
#
# **Task**
# 1. Run the 10-line spam model on the *SMS Spam Collection* dataset and note its accuracy.
# 2. Write messages that try to fool it (spam that slips through, or normal messages that get flagged).
# 3. Record what it predicted and why it failed.
#
# **Dataset:** SMS Spam Collection (5,574 labelled text messages) - https://archive.ics.uci.edu/dataset/228/sms+spam+collection
# The dataset is **downloaded when you run this file; it is not stored in the repository.**
#
# Run as a script: `python spam_filter.py`  |  Run in Jupyter/Colab: run the cells top to bottom.

# %% [markdown]
# ## 0. Setup

# %%
#NB# %pip install -q pandas scikit-learn

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Works both as a script (uses __file__) and inside a notebook (uses the current folder)
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

DATASET_PAGE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
LOCAL_FILES = [BASE_DIR / "sms.tsv", BASE_DIR / "SMSSpamCollection"]  # both are "label<TAB>message" files (git-ignored)

# %% [markdown]
# ## 1. Load the dataset and train the 10-line model

# %%
def load_sms_dataset():
    """Use a local copy if there is one, otherwise download it. Columns: label (ham/spam), text."""
    for path in LOCAL_FILES:
        if path.exists():
            print(f"Using local file {path.name}")
            return pd.read_csv(path, sep="\t", names=["label", "text"])
    try:
        df = pd.read_csv(DATASET_URL, sep="\t", names=["label", "text"])
        df.to_csv(LOCAL_FILES[0], sep="\t", header=False, index=False)  # cache it (git-ignored)
        print("Downloaded the dataset")
        return df
    except Exception as error:
        raise RuntimeError(
            "Could not download the dataset. Download it from "
            f"{DATASET_PAGE}, unzip it and put the file 'SMSSpamCollection' next to this script."
        ) from error


df = load_sms_dataset()
print(df["label"].value_counts().to_string())
print()
print(df.head(3).to_string())

# %%
# ---- the 10-line model -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
vec = CountVectorizer()                      # text -> word counts
model = MultinomialNB()                      # Naive Bayes classifier
model.fit(vec.fit_transform(X_train), y_train)
accuracy = model.score(vec.transform(X_test), y_test)
print(f"Accuracy on the test set: {accuracy:.4f} ({accuracy * 100:.2f}%)")
# -------------------------------------------------------------------------------------------

# %% [markdown]
# ### Is that accuracy actually good?
# Only about 13% of the messages are spam, so a "model" that always says *ham* would already score about 87%.
# The confusion matrix shows where the real model makes mistakes.

# %%
baseline = (y_test == "ham").mean()
print(f"Baseline (always say 'ham'): {baseline:.4f}")

predictions = model.predict(vec.transform(X_test))
tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=["ham", "spam"]).ravel()
print(f"Correct ham: {tn} | ham wrongly flagged as spam: {fp} | spam that slipped through: {fn} | correct spam: {tp}")
print(f"Spam caught (recall): {tp / (tp + fn):.3f} | flagged messages that really are spam (precision): {tp / (tp + fp):.3f}")

# %% [markdown]
# ### What does the model "look at"?
# Naive Bayes learns how much more likely each *word* is in spam than in normal messages.
# The words below are its strongest spam signals - and the model knows **nothing** about words it has never seen.

# %%
def top_spam_words(model, vec, n=12):
    """Words with the highest log-ratio P(word | spam) / P(word | ham)."""
    spam_i, ham_i = list(model.classes_).index("spam"), list(model.classes_).index("ham")
    ratio = model.feature_log_prob_[spam_i] - model.feature_log_prob_[ham_i]
    words = np.array(vec.get_feature_names_out())
    return list(words[np.argsort(-ratio)[:n]])


print("Strongest spam words:", ", ".join(top_spam_words(model, vec)))

# %% [markdown]
# ## 2. Try to fool the model
# Each message has a *trick* (the idea behind the attack), the *correct* label, and my *hypothesis*
# about why a bag-of-words model could fail. The hypothesis is only shown in the table if the model really was fooled.

# %%
attacks = [
    {"text": "Cl@im your fr33 g1ft card now", "should_be": "spam",
     "trick": "Spelling trick (leetspeak)",
     "why": "Spelling trick: 'cl@im', 'fr33' and 'g1ft' are words the model has never seen, so they carry no spam evidence"},
    {"text": "Not free, just confirming lunch at 1pm", "should_be": "ham",
     "trick": "Negation",
     "why": "Saw 'free' (a strong spam word) and ignored the 'not' before it: word counts have no sense of order"},
    {"text": "Dear customer, your parcel could not be delivered. Please confirm your delivery address at http://bit.ly/3xYz to avoid return.",
     "should_be": "spam", "trick": "Polite, businesslike tone",
     "why": "Sounds like a normal business message: none of the loud spam words (WIN, FREE, CLAIM) appear"},
    {"text": "F R E E entry in our weekly draw, text W I N to 80082", "should_be": "spam",
     "trick": "Spaced-out letters",
     "why": "Single letters are split into separate one-letter tokens that the model ignores, so the spam words vanish"},
    {"text": "Congratulations! You won the Software Engineering hackathon. Your prize can be collected tomorrow at the main office.",
     "should_be": "ham", "trick": "Spam words in a real message",
     "why": "'Congratulations', 'won' and 'prize' are strong spam words in the training data, but here they are innocent"},
    {"text": "Call me when you are free, I have a free ticket for you", "should_be": "ham",
     "trick": "Innocent use of 'free'",
     "why": "'free' appears twice and is mostly seen in spam, so the model ignores the friendly context"},
    {"text": "Your bank account needs verificatiоn, please log in at the link below", "should_be": "spam",
     "trick": "Look-alike characters (Cyrillic 'o')",
     "why": "The Cyrillic 'o' makes 'verification' a brand-new unknown word, so it is not recognised"},
    {"text": "check this out http://bit.ly/2xYz", "should_be": "spam",
     "trick": "Very short, link only",
     "why": "Too little text: almost no known spam words to add up"},
    {"text": "URGENT! You have been selected for a cash prize of 1000. Call 09061701461 now to claim",
     "should_be": "spam", "trick": "Control: classic spam (should be caught)",
     "why": "Control message - classic spam wording"},
]

# %% [markdown]
# ## 3. Record what it predicted and why it failed

# %%
def predict_messages(model, vec, texts):
    """Return the predicted label and the spam probability for each message."""
    matrix = vec.transform(texts)
    spam_col = list(model.classes_).index("spam")
    return model.predict(matrix), model.predict_proba(matrix)[:, spam_col]


def build_results_table(attacks, predicted, spam_probability):
    rows = []
    for attack, pred, prob in zip(attacks, predicted, spam_probability):
        fooled = pred != attack["should_be"]
        rows.append({
            "My message": attack["text"],
            "Trick": attack["trick"],
            "Model said": pred.replace("ham", "not spam"),
            "Should be": attack["should_be"].replace("ham", "not spam"),
            "Spam prob.": f"{prob:.2f}",
            "Fooled?": "YES" if fooled else "no",
            "Why it failed": attack["why"] if fooled else "- (model got it right)",
        })
    return pd.DataFrame(rows)


def to_markdown(table):
    """Make a Markdown table (for the README) without extra libraries."""
    header = "| " + " | ".join(table.columns) + " |"
    divider = "|" + "|".join(["---"] * len(table.columns)) + "|"
    lines = [header, divider]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)


texts = [a["text"] for a in attacks]
predicted, spam_probability = predict_messages(model, vec, texts)
results = build_results_table(attacks, predicted, spam_probability)

# The exact code from the task slide:
print(model.predict(vec.transform(texts[:2])))
print()
print(results[["My message", "Model said", "Should be", "Spam prob.", "Fooled?"]].to_string(index=False))

attack_only = results[~results["Trick"].str.startswith("Control")]
print()
print(f"The model was fooled by {(attack_only['Fooled?'] == 'YES').sum()} of {len(attack_only)} attack messages.")

# %% [markdown]
# ### Results table for the README
# Copy this table into `README.md` (it is also saved to `results_table.md`).

# %%
table_md = to_markdown(results)
(BASE_DIR / "results_table.md").write_text(table_md + "\n", encoding="utf-8")
print(table_md)

# %% [markdown]
# ## 4. Extra: can a different model resist the tricks?
# Model v2 reads **groups of characters** (character n-grams) instead of whole words, so "fr33" still looks a bit like "free",
# and it uses Logistic Regression instead of Naive Bayes. Same data, same split, same attack messages.

# %%
vec2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True)
model2 = LogisticRegression(max_iter=2000, class_weight="balanced")
model2.fit(vec2.fit_transform(X_train), y_train)
accuracy2 = model2.score(vec2.transform(X_test), y_test)
print(f"v1 (words + Naive Bayes) accuracy:          {accuracy:.4f}")
print(f"v2 (character n-grams + LogReg) accuracy: {accuracy2:.4f}")

predicted2, _ = predict_messages(model2, vec2, texts)
compare = pd.DataFrame({
    "Message": [t if len(t) <= 55 else t[:52] + "..." for t in texts],
    "Should be": [a["should_be"] for a in attacks],
    "v1 said": predicted,
    "v2 said": predicted2,
})
print()
print(compare.to_string(index=False))
fooled_v1 = int((predicted != np.array([a["should_be"] for a in attacks])).sum())
fooled_v2 = int((predicted2 != np.array([a["should_be"] for a in attacks])).sum())
print(f"\nMessages wrong: v1 = {fooled_v1} of {len(texts)}, v2 = {fooled_v2} of {len(texts)}")

# %% [markdown]
# ## 5. What I learned
# Write your own 2 sentences here after you have seen your results (see the README for the template).
