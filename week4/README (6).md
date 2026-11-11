# Week 4 - Prompt-Tester (Small AI App)

An LLM-powered app that **compares prompt versions against a golden set**.

**Use case:** an LLM triages university IT-support tickets. For every ticket it must return validated JSON:

```json
{"category": "network", "priority": "high", "summary": "Lab computers have no internet during an exam", "needs_human": true}
```

The app runs three prompt versions (`v1_basic`, `v2_structured`, `v3_fewshot`) on 14 tickets with known answers
and shows which prompt works best.

## What is in this folder

| File | What it is |
|------|------------|
| `prompt_tester.py` | The app (script) |
| `prompt_tester.ipynb` | Same code as a Colab/Jupyter notebook |
| `prompts/v1_basic.txt`, `v2_structured.txt`, `v3_fewshot.txt` | The prompt versions (the prompt lives in files, not in code) |
| `data/golden_set.json` | 14 tickets with the expected category, priority, `needs_human` and summary keywords |
| `results/` | Created by a run: `results.md` (comparison), `examples.md` (good and bad output), `details.json` |

## How to run

### 1. Get an API key (free)

- **Gemini:** create a free key in [Google AI Studio](https://aistudio.google.com/apikey).
- or **Anthropic:** a key from the Anthropic Console.

Keys are read from **environment variables** and are never written in the code.

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key"            # or ANTHROPIC_API_KEY
# Windows PowerShell
$env:GEMINI_API_KEY="your-key"
```

You can also put `GEMINI_API_KEY=your-key` in a file called `.env` next to the script (it is git-ignored).

Optional settings: `LLM_PROVIDER` (`gemini`, `anthropic`, `mock`), `LLM_MODEL` (change it if the default model is retired),
`CALL_DELAY` (seconds between calls; default 4 for Gemini to respect the free rate limit).

### 2. Run locally

```bash
# from the repository root
python -m venv venv
#   Windows (PowerShell): venv\Scripts\Activate.ps1   |   macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
cd week-04
python prompt_tester.py
```

A full run makes 42 API calls (14 tickets x 3 prompts) and takes about 3 minutes with a free Gemini key.

**No key?** The app falls back to **MOCK mode** (a keyword rule, not an LLM) so you can test the pipeline.
In mock mode all prompt versions give identical answers, so the comparison is not meaningful.

### 3. Run in Google Colab

1. **File -> Upload notebook** -> `prompt_tester.ipynb`.
2. Click the key icon in the left sidebar (**Secrets**), add `GEMINI_API_KEY`, and switch on notebook access.
3. **Runtime -> Run all**. When asked, select the 4 files `v1_basic.txt`, `v2_structured.txt`, `v3_fewshot.txt`, `golden_set.json`.
4. Download the executed notebook and commit it, together with the `results/` files.

## Architecture / pipeline

```
 prompts/v*.txt ---+
                   |  fill {ticket}
 golden_set.json --+------> call_llm() ------> parse_json() ------> Ticket (pydantic) ---- valid ----> score vs golden answers
                              ^                                           |                                   |
                              |                                           | invalid JSON / wrong schema       v
                              +------ retry (max 3 attempts) <------------+                          results/*.md, details.json
                              |        with a "repair note"
                              +------ back-off (3s, 6s) if the API is busy or rate limited
```

1. **Load**: prompt versions from files, test cases from the golden set.
2. **Augment**: the ticket text is inserted into the prompt template.
3. **Generate**: `call_llm` sends it to Gemini or Anthropic (temperature 0).
4. **Validate**: the reply must parse as JSON and match the `Ticket` schema (allowed values, summary length, real boolean).
5. **Retry**: bad replies are re-asked with a note saying what was wrong; temporary API errors wait and retry; fatal errors (bad key, wrong model) stop the run.
6. **Evaluate**: each answer is compared to the golden answer and the results are saved.

## Prompt design choices

| Version | Idea | Why |
|---------|------|-----|
| `v1_basic` | One sentence asking for JSON with four keys | Baseline: what happens with a lazy prompt? |
| `v2_structured` | Role, exact schema with allowed values, definitions for each category, rules for priority and `needs_human`, "ONLY JSON" | Removes ambiguity so the answers can be validated |
| `v3_fewshot` | `v2` + three worked examples | Examples show the exact output style; they are *not* taken from the golden set (no leakage) |

## Example outputs

**Good output** (accepted by the validator):

```json
{"category": "network", "priority": "low", "summary": "Wi-Fi is slow in the dorm", "needs_human": false}
```

**Bad outputs** (real validator messages from the app's built-in demo; each of these triggers a retry):

| Reply from the model | Validator result |
|----------------------|------------------|
| `Sure! This looks like a network problem with low priority.` | rejected: the reply contains no JSON object |
| `{"category": "Network issue", ...}` | rejected: `category: Input should be 'account_access', 'network', 'software', 'hardware' or 'other'` |
| `{"needs_human": "maybe", ...}` | rejected: `needs_human: Input should be a valid boolean, unable to interpret input` |

A run also writes `results/examples.md` with one good and one bad output from the best prompt on the real golden set.

## Results (from my run)

Run the app and paste the table from `results/results.md` here:

| version | valid_% | first_try_% | category_% | priority_% | needs_human_% | summary_% | score_% |
|---------|---------|-------------|------------|------------|---------------|-----------|---------|
| v1_basic | | | | | | | |
| v2_structured | | | | | | | |
| v3_fewshot | | | | | | | |

## How it is evaluated

- **Golden tests:** 14 tickets with expected `category`, `priority`, `needs_human` and at least one expected keyword in the summary.
- **Rubric (per ticket):** valid JSON after at most 3 attempts; valid on the first try; category, priority and `needs_human` equal to the expected values; summary contains an expected keyword.
  `score_%` is the average of the last four checks. An invalid reply counts as wrong everywhere.
- **Offline self-tests:** 6 tests with fake LLMs prove that retries, back-off, repair notes and fatal errors behave correctly (no API key needed).
- **Mistake list:** the app prints every wrong field so you can see why a version lost points.
- **Next step (not done):** an LLM-as-judge prompt to grade summary quality.

## Limitations and responsible-AI notes

- **Small, synthetic golden set** (14 tickets I wrote): scores are indicative, not proof. Real tickets are messier and longer.
- **LLM answers can vary** between runs even at temperature 0, and models get retired, so results can change.
- **Summary check is crude:** a keyword match does not measure whether the summary is truly good.
- **Mock mode is not an AI:** its output only tests the pipeline.
- **Privacy:** real tickets can contain names, IDs and passwords. Do not send real student data to a third-party API without permission; remove personal data first.
- **Safety:** a wrong `priority` or `needs_human` on a security ticket could delay help, so a human should review high-priority and security tickets.
- **Secrets:** API keys come from environment variables or a git-ignored `.env`; never commit them.
- **Language:** tickets are in English only; other languages were not tested.

## What I learned

- A prompt is code: keep it in a file, version it, and measure every change on the same golden set.
- Validation and retries make an unreliable text generator usable inside a program.
- Fake LLMs in self-tests let me test the retry logic without spending API calls.
