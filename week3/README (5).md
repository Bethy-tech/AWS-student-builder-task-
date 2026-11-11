# Week 3 - Break the Spam Filter

I train a tiny spam filter on the **SMS Spam Collection** dataset, then write messages that try to **fool it**
(spam that slips through, or normal messages that get flagged) and record why it failed.

- **Dataset:** [SMS Spam Collection (UCI)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) - 5,574 labelled messages.
  The dataset is **not uploaded to this repository**; the script downloads it when you run it (the copy is git-ignored).
- **Model:** `CountVectorizer` (word counts) + `MultinomialNB` (Naive Bayes), 80/20 train/test split with `random_state=42`.

## Files

| File | What it is |
|------|------------|
| `spam_filter.py` | The whole experiment as a Python script |
| `spam_filter.ipynb` | Same code as a Colab/Jupyter notebook (commit it **after running it** so the outputs are saved) |
| `results_table.md` | Results table, created automatically when you run the script/notebook |

## How to run

### Locally (virtual environment)

```bash
# from the repository root
python -m venv venv
#   Windows (PowerShell): venv\Scripts\Activate.ps1   |   macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
cd week-03
python spam_filter.py
```

If the automatic download is blocked, download the zip from the dataset link, unzip it and put the file
`SMSSpamCollection` next to `spam_filter.py`.

### In Google Colab

**File -> Upload notebook** -> `spam_filter.ipynb` -> **Runtime -> Run all** -> **File -> Download -> Download .ipynb**.

## Step 1 - Accuracy of the 10-line model

Run the notebook and write the numbers here:

- Accuracy on the test set: **`____ %`** (the script prints it)
- A model that always says "ham" would already get about 87%, so the script also prints the confusion matrix,
  the share of spam caught (recall) and precision.

## Step 2 and 3 - Messages that try to fool the model

I used 8 attack messages with different tricks, plus 1 control (a classic spam message the model *should* catch).
The notebook fills in what the model predicted, and shows my explanation **only when the model was really fooled**.
Copy the final table from `results_table.md` (it has the same rows plus the spam probability).

| My message | Model said | Should be | Why it failed |
|------------|------------|-----------|---------------|
| `Cl@im your fr33 g1ft card now` | _from notebook_ | spam | Spelling trick: words the model has never seen |
| `Not free, just confirming lunch at 1pm` | _from notebook_ | not spam | Negation: sees "free", ignores "not" |
| `Dear customer, your parcel could not be delivered. Please confirm your delivery address at http://bit.ly/3xYz to avoid return.` | _from notebook_ | spam | Polite, businesslike tone: no loud spam words |
| `F R E E entry in our weekly draw, text W I N to 80082` | _from notebook_ | spam | Spaced-out letters become ignored one-letter tokens |
| `Congratulations! You won the Software Engineering hackathon. Your prize can be collected tomorrow at the main office.` | _from notebook_ | not spam | Spam words used in a normal message |
| `Call me when you are free, I have a free ticket for you` | _from notebook_ | not spam | Innocent use of "free" |
| `Your bank account needs verificati`**`о`**`n, please log in at the link below` | _from notebook_ | spam | Look-alike (Cyrillic) letter makes a new unknown word |
| `check this out http://bit.ly/2xYz` | _from notebook_ | spam | Too short: almost no known spam words |

The "Why it failed" column is my hypothesis; it only counts for the messages where the model really got it wrong
(the notebook marks those with `Fooled? YES`).

### Tricks used (3 or more required)

1. **Spelling tricks / look-alike characters** (`fr33`, `cl@im`, Cyrillic `о`)
2. **Negation** (`Not free`)
3. **Polite, businesslike tone** (phishing that sounds normal)
4. **Spaced-out letters** (`F R E E`)
5. **Spam words in a real message** (`Congratulations`, `prize`, `free`)
6. **Very short, link-only messages**

## Extra - can a different model resist?

The notebook also trains **v2**: character n-grams (`TfidfVectorizer(analyzer="char_wb")`) + Logistic Regression,
on the same split, and compares both models on the same messages.

## What I learned

_Write your own two sentences after you have looked at your results. Template:_

> The model fails when a message _____ because it only counts words and ignores _____.
> Next time I would _____ (for example, use character n-grams or look at word order) to make it more robust.
