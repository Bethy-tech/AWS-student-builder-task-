# %% [markdown]
# # Week 4 - Prompt-Tester: compare prompt versions against a golden set
#
# **Use case:** an LLM triages university IT-support tickets. For every ticket it must return validated JSON:
# `category`, `priority`, `summary`, `needs_human`.
#
# **What this app does**
# 1. Loads three prompt versions from files (`prompts/`) and a golden set of 14 tickets with known answers (`data/golden_set.json`).
# 2. Calls an LLM API for every ticket with every prompt version.
# 3. Validates each reply (JSON + schema), retries when the reply is bad or the API is busy.
# 4. Scores each prompt version against the golden answers and saves the results to `results/`.
#
# Run as a script: `python prompt_tester.py`  |  Run in Jupyter/Colab: run the cells top to bottom.
#
# **API keys:** set `GEMINI_API_KEY` (free key from Google AI Studio) *or* `ANTHROPIC_API_KEY` as an environment variable.
# With no key the app runs in **MOCK mode** (a simple keyword rule instead of an LLM) so the pipeline can still be tested.

# %% [markdown]
# ## 0. Setup

# %%
#NB# %pip install -q requests pydantic python-dotenv pandas

# %%
import json
import os
import re
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError

try:  # optional: read keys from a local .env file (never commit it!)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Works both as a script (uses __file__) and inside a notebook (uses the current folder)
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROMPT_DIR = BASE_DIR / "prompts"
GOLDEN_PATH = BASE_DIR / "data" / "golden_set.json"
RESULTS_DIR = BASE_DIR / "results"

#NB# # In Colab the repo files are not there by default: upload the 4 project files when asked.
#NB# needed = ["prompts/v1_basic.txt", "prompts/v2_structured.txt", "prompts/v3_fewshot.txt", "data/golden_set.json"]
#NB# if not all((BASE_DIR / p).exists() for p in needed):
#NB#     from google.colab import files
#NB#     print("Select v1_basic.txt, v2_structured.txt, v3_fewshot.txt and golden_set.json")
#NB#     for name, content in files.upload().items():
#NB#         target = BASE_DIR / ("data" if name.endswith(".json") else "prompts") / name
#NB#         target.parent.mkdir(exist_ok=True)
#NB#         target.write_bytes(content)
#NB#
#NB# # Colab: store your key in the left sidebar (key icon -> Secrets) as GEMINI_API_KEY or ANTHROPIC_API_KEY
#NB# try:
#NB#     from google.colab import userdata
#NB#     for key_name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
#NB#         try:
#NB#             os.environ[key_name] = userdata.get(key_name)
#NB#         except Exception:
#NB#             pass
#NB# except ImportError:
#NB#     pass

MAX_ATTEMPTS = 3        # tries per ticket (first try + retries)
BACKOFF_SECONDS = 3.0   # wait 3s, 6s, 12s ... after a busy/failed API call
DEFAULT_MODELS = {"gemini": "gemini-flash-latest", "anthropic": "claude-haiku-5-5"}  # change with LLM_MODEL

# %% [markdown]
# ## 1. Prompts live in files
# Keeping prompts in text files (not inside the code) makes them easy to edit, compare and track in Git.
# `{ticket}` is replaced by the ticket text.

# %%
def load_prompts(folder):
    """Read every prompt file: {'v1_basic': '...', 'v2_structured': '...', ...} (sorted by name)."""
    return {path.stem: path.read_text(encoding="utf-8") for path in sorted(folder.glob("v*.txt"))}


prompts = load_prompts(PROMPT_DIR)
print("Prompt versions:", ", ".join(prompts))
print()
print("--- v1_basic ---")
print(prompts["v1_basic"])

# %% [markdown]
# ## 2. The LLM client
# One function (`call_llm`) hides the provider. It raises `LLMError` for problems worth retrying (busy API, timeout)
# and `FatalLLMError` for problems that retrying cannot fix (missing key, wrong model name).

# %%
class LLMError(Exception):
    """Temporary problem (rate limit, server error, timeout): retry later."""


class FatalLLMError(Exception):
    """Permanent problem (missing/invalid key, wrong model): stop and tell the user."""


def get_provider():
    """Pick the provider: LLM_PROVIDER if set, otherwise whichever API key exists, otherwise mock."""
    forced = os.getenv("LLM_PROVIDER", "").lower()
    if forced in ("gemini", "anthropic", "mock"):
        return forced
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "mock"


def get_model(provider):
    return os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, "mock")


def _check_http(response):
    """Turn HTTP errors into LLMError (retry) or FatalLLMError (do not retry)."""
    if response.status_code in (429, 500, 502, 503, 504):
        raise LLMError(f"HTTP {response.status_code} (busy or rate limited)")
    if response.status_code >= 400:
        raise FatalLLMError(f"HTTP {response.status_code}: {response.text[:300]}")


def call_gemini(prompt, model):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise FatalLLMError("GEMINI_API_KEY is not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    response = requests.post(
        url,
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}},
        timeout=60,
    )
    _check_http(response)
    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as error:
        raise LLMError("empty or blocked reply") from error


def call_anthropic(prompt, model):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise FatalLLMError("ANTHROPIC_API_KEY is not set")
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": model, "max_tokens": 400, "temperature": 0, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    _check_http(response)
    try:
        return response.json()["content"][0]["text"]
    except (KeyError, IndexError, TypeError) as error:
        raise LLMError("empty reply") from error


def call_mock(prompt):
    """NOT an LLM: a keyword rule that returns JSON so the pipeline can be tested without a key."""
    ticket = prompt.rsplit("Ticket:", 1)[-1].split("\n\n")[0].strip()
    text = ticket.lower()

    def has(*words):
        return any(w in text for w in words)

    if has("password", "log in", "login", "logged into", "username", "mailbox"):
        category = "account_access"
    elif has("wi-fi", "wifi", "internet", "vpn", "network"):
        category = "network"
    elif has("install", "licence", "license", "matlab", "python", "visual studio", "freez", "software"):
        category = "software"
    elif has("laptop", "screen", "printer", "projector", "computer"):
        category = "hardware"
    else:
        category = "other"
    priority = "high" if has("exam", "deadline", "whole", "someone else", "10 minutes") else (
        "low" if has("how do i", "any tips", "what time", "when ", "hello", "forgot my username", "out of paper") else "medium")
    needs_human = category == "hardware" or has("licence", "unknown location", "never arrives", "full", "someone else")
    return json.dumps({"category": category, "priority": priority,
                       "summary": " ".join(ticket.split()[:14]), "needs_human": needs_human})


def call_llm(prompt):
    """Send one prompt to the chosen provider and return the raw text reply."""
    provider = get_provider()
    model = get_model(provider)
    try:
        if provider == "gemini":
            return call_gemini(prompt, model)
        if provider == "anthropic":
            return call_anthropic(prompt, model)
        return call_mock(prompt)
    except requests.RequestException as error:  # timeouts, connection problems
        raise LLMError(f"network problem: {error}") from error


provider = get_provider()
print(f"Provider: {provider} (model: {get_model(provider)})")
if provider == "mock":
    print("MOCK MODE: no API key found, so a simple keyword rule answers instead of an LLM.")
    print("Every prompt version gets the same answers, so the comparison is NOT meaningful. Set an API key for real results.")

# %% [markdown]
# ## 3. Structured output + validation
# The reply must be a JSON object that matches the `Ticket` schema (allowed values, summary length, real `true/false`).
# Anything else is rejected, and the retry step asks the model again.

# %%
class Ticket(BaseModel):
    category: Literal["account_access", "network", "software", "hardware", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str = Field(min_length=5, max_length=200)
    needs_human: bool


def parse_json(raw):
    """Pull the JSON object out of a reply, even if the model added ```json fences or some chatter."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("the reply contains no JSON object")
    return json.loads(text[start:end + 1])


def describe_error(error):
    """Short, readable error text (also sent back to the model when we retry)."""
    if isinstance(error, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return str(error)


# Quick demo of what the validator accepts and rejects
good = '{"category": "network", "priority": "low", "summary": "Wi-Fi is slow in the dorm", "needs_human": false}'
bad_examples = {
    "prose instead of JSON": "Sure! This looks like a network problem with low priority.",
    "category not allowed": '{"category": "Network issue", "priority": "low", "summary": "Wi-Fi is slow", "needs_human": false}',
    "needs_human is text": '{"category": "network", "priority": "low", "summary": "Wi-Fi is slow", "needs_human": "maybe"}',
}
print("GOOD ->", Ticket.model_validate(parse_json(good)).model_dump())
for label, reply in bad_examples.items():
    try:
        Ticket.model_validate(parse_json(reply))
        print(f"BAD ({label}) -> unexpectedly accepted")
    except ValueError as error:  # JSON errors and pydantic ValidationError are both ValueErrors
        print(f"BAD ({label}) -> rejected: {describe_error(error)}")

# %% [markdown]
# ## 4. Error handling and retries
# `run_prompt` tries up to `MAX_ATTEMPTS` times:
# - **API busy / timeout** (`LLMError`) -> wait with exponential back-off, then try again.
# - **Bad reply** (not JSON, wrong schema) -> ask again and tell the model what was wrong.
# - **Fatal problem** (`FatalLLMError`) -> stop immediately with a clear message.

# %%
REPAIR_NOTE = "\n\nYour previous reply was rejected: {error}\nReply again with ONLY one valid JSON object that follows the schema."


def run_prompt(template, ticket_text, llm=None, max_attempts=MAX_ATTEMPTS, wait=time.sleep):
    """Run one ticket through one prompt template. Returns a dict with the validated ticket or the last error."""
    llm = llm or call_llm
    prompt = template.replace("{ticket}", ticket_text)
    last_error, raw = "", ""
    for attempt in range(1, max_attempts + 1):
        try:
            raw = llm(prompt if attempt == 1 else prompt + REPAIR_NOTE.format(error=last_error))
            ticket = Ticket.model_validate(parse_json(raw))
            return {"ok": True, "ticket": ticket.model_dump(), "attempts": attempt, "raw": raw, "error": ""}
        except LLMError as error:  # temporary API problem -> back off and retry
            last_error = str(error)
            if attempt < max_attempts:
                wait(BACKOFF_SECONDS * 2 ** (attempt - 1))
        except ValueError as error:  # bad JSON or schema mismatch -> retry with a repair note
            last_error = describe_error(error)
    return {"ok": False, "ticket": None, "attempts": max_attempts, "raw": raw, "error": last_error}

# %% [markdown]
# ### Offline self-tests
# These tests use fake "LLMs", so they need no API key and no internet. They prove the retry logic works.

# %%
def run_self_tests():
    template = "Ticket: {ticket}"
    ok_reply = '{"category": "other", "priority": "low", "summary": "A simple question", "needs_human": false}'

    # 1) fenced JSON with extra chatter is still accepted on the first try
    result = run_prompt(template, "x", llm=lambda p: "Here you go:\n```json\n" + ok_reply + "\n```")
    assert result["ok"] and result["attempts"] == 1

    # 2) prose first, valid JSON second -> succeeds on attempt 2 and the repair note was sent
    prompts_seen = []
    replies = iter(["I think it is a network problem.", ok_reply])
    result = run_prompt(template, "x", llm=lambda p: (prompts_seen.append(p), next(replies))[1])
    assert result["ok"] and result["attempts"] == 2 and "rejected" in prompts_seen[1]

    # 3) schema violation (category not allowed) is rejected, then fixed
    bad = ok_reply.replace('"other"', '"misc"')
    replies = iter([bad, ok_reply])
    result = run_prompt(template, "x", llm=lambda p: next(replies))
    assert result["ok"] and result["attempts"] == 2

    # 4) two temporary API errors, then success; back-off waits are 3s and 6s (not really slept here)
    calls, waits = {"n": 0}, []

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMError("HTTP 429 (busy or rate limited)")
        return ok_reply

    result = run_prompt(template, "x", llm=flaky, wait=waits.append)
    assert result["ok"] and result["attempts"] == 3 and waits == [3.0, 6.0]

    # 5) always garbage -> gives up after MAX_ATTEMPTS with a helpful error
    result = run_prompt(template, "x", llm=lambda p: "no json here", wait=lambda s: None)
    assert not result["ok"] and result["attempts"] == MAX_ATTEMPTS and "no JSON" in result["error"]

    # 6) fatal errors are NOT retried: they stop the run
    def fatal(prompt):
        raise FatalLLMError("HTTP 404: model not found")

    try:
        run_prompt(template, "x", llm=fatal)
        raise AssertionError("FatalLLMError should have stopped the run")
    except FatalLLMError:
        pass
    return 6


print(f"All {run_self_tests()} self-tests passed")

# %% [markdown]
# ## 5. Golden set + evaluation
# The **golden set** has 14 tickets with the answer a human expert expects. Scoring rubric (per ticket):
#
# | Check | Passes when |
# |-------|-------------|
# | valid | a valid JSON reply was obtained (after at most 3 attempts) |
# | first try | it was valid on the first attempt (no retry needed) |
# | category | equals the expected category |
# | priority | equals the expected priority |
# | needs_human | equals the expected value |
# | summary | the summary contains at least one expected keyword |
#
# An invalid reply counts as wrong for every check. `score` is the average of category, priority, needs_human and summary.

# %%
def load_golden_set(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_prompt(version, template, cases, llm=None, wait=time.sleep, delay=0.0):
    """Run every golden case through one prompt version. Returns one row per case."""
    rows = []
    for case in cases:
        started = time.time()
        result = run_prompt(template, case["text"], llm=llm, wait=wait)
        seconds = time.time() - started
        row = {"version": version, "case_id": case["id"], "valid": result["ok"], "first_try": result["ok"] and result["attempts"] == 1,
               "attempts": result["attempts"], "seconds": round(seconds, 2), "error": result["error"],
               "category_ok": False, "priority_ok": False, "needs_human_ok": False, "summary_ok": False, "predicted": None}
        if result["ok"]:
            predicted, expected = result["ticket"], case["expected"]
            row["category_ok"] = predicted["category"] == expected["category"]
            row["priority_ok"] = predicted["priority"] == expected["priority"]
            row["needs_human_ok"] = predicted["needs_human"] == expected["needs_human"]
            row["summary_ok"] = any(k in predicted["summary"].lower() for k in case["summary_keywords"])
            row["predicted"] = predicted
        rows.append(row)
        if delay:
            time.sleep(delay)  # stay under free-tier rate limits
    return pd.DataFrame(rows)


def summarize(details):
    """One row per prompt version with percentages."""
    grouped = details.groupby("version", sort=False)
    table = pd.DataFrame({
        "cases": grouped.size(),
        "valid_%": grouped["valid"].mean() * 100,
        "first_try_%": grouped["first_try"].mean() * 100,
        "category_%": grouped["category_ok"].mean() * 100,
        "priority_%": grouped["priority_ok"].mean() * 100,
        "needs_human_%": grouped["needs_human_ok"].mean() * 100,
        "summary_%": grouped["summary_ok"].mean() * 100,
        "avg_attempts": grouped["attempts"].mean(),
        "avg_seconds": grouped["seconds"].mean(),
    })
    table["score_%"] = table[["category_%", "priority_%", "needs_human_%", "summary_%"]].mean(axis=1)
    return table.round(1).reset_index()


golden = load_golden_set(GOLDEN_PATH)
print(f"Golden set: {len(golden)} tickets, {len(prompts)} prompt versions -> {len(golden) * len(prompts)} LLM calls")

# %% [markdown]
# ## 6. Run the comparison

# %%
delay = float(os.getenv("CALL_DELAY", "4" if provider == "gemini" else "0"))  # free Gemini keys allow few calls per minute
all_rows = []
try:
    for version, template in prompts.items():
        print(f"Testing {version} ...")
        all_rows.append(evaluate_prompt(version, template, golden, delay=delay))
except FatalLLMError as error:
    raise SystemExit(f"Stopped: {error}\nCheck your API key / LLM_MODEL (see the README).")

details = pd.concat(all_rows, ignore_index=True)
summary = summarize(details)
print()
print(summary.to_string(index=False))
best = summary.sort_values(["score_%", "first_try_%"], ascending=False).iloc[0]["version"]
print(f"\nBest prompt version: {best}")

# %% [markdown]
# ## 7. Where did each version go wrong?
# One row per mistake, so we can see *why* a version lost points.

# %%
def list_mistakes(details, golden):
    texts = {c["id"]: c for c in golden}
    lines = []
    for row in details.itertuples():
        if row.valid:
            expected, predicted = texts[row.case_id]["expected"], row.predicted
            for field in ("category", "priority", "needs_human"):
                if predicted[field] != expected[field]:
                    lines.append(f"{row.version} | {row.case_id} | {field}: expected {expected[field]!r}, got {predicted[field]!r}")
            if not row.summary_ok:
                lines.append(f"{row.version} | {row.case_id} | summary has none of {texts[row.case_id]['summary_keywords']}: {predicted['summary']!r}")
        else:
            lines.append(f"{row.version} | {row.case_id} | INVALID after {row.attempts} attempts: {row.error}")
    return lines


mistakes = list_mistakes(details, golden)
print(f"{len(mistakes)} mistake(s) in total")
for line in mistakes[:25]:
    print(" -", line)
if len(mistakes) > 25:
    print(f" ... and {len(mistakes) - 25} more (see results/details.json)")

# %% [markdown]
# ## 8. Save the results (commit them to Git)
# - `results/results.md` - comparison table
# - `results/examples.md` - a good and a bad example output from the best version
# - `results/details.json` - every case, every version

# %%
def to_markdown(table):
    header = "| " + " | ".join(table.columns) + " |"
    lines = [header, "|" + "|".join(["---"] * len(table.columns)) + "|"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def build_examples(details, golden, best):
    """A good output and a bad output (if there is one) from the best version, as Markdown."""
    texts = {c["id"]: c for c in golden}
    rows = details[details["version"] == best]
    perfect = rows[rows["valid"] & rows["category_ok"] & rows["priority_ok"] & rows["needs_human_ok"] & rows["summary_ok"]]
    flawed = rows[~rows.index.isin(perfect.index)]
    parts = [f"# Example outputs from `{best}`\n"]
    for title, subset in (("Good example", perfect), ("Bad example", flawed)):
        if subset.empty:
            note = "no ticket passed every check." if title.startswith("Good") else "every ticket passed every check."
            parts.append(f"## {title}\n\nNone: {note}\n")
            continue
        row = subset.iloc[0]
        case = texts[row["case_id"]]
        parts.append(f"## {title} ({row['case_id']})\n\n**Ticket:** {case['text']}\n\n**Expected:** `{json.dumps(case['expected'])}`\n\n"
                     f"**Model output:** `{json.dumps(row['predicted']) if row['valid'] else 'INVALID - ' + row['error']}`\n")
    return "\n".join(parts)


RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "results.md").write_text(
    f"# Prompt comparison (provider: {provider}, model: {get_model(provider)})\n\n{to_markdown(summary)}\n", encoding="utf-8")
(RESULTS_DIR / "examples.md").write_text(build_examples(details, golden, best), encoding="utf-8")
details.to_json(RESULTS_DIR / "details.json", orient="records", indent=2)
print("Saved results.md, examples.md and details.json in the results/ folder")
print()
print((RESULTS_DIR / "examples.md").read_text(encoding="utf-8"))

# %% [markdown]
# ## 9. What I learned
# - A prompt is code: keep it in a file, version it (v1, v2, v3) and measure every change against the same golden set.
# - Validation + retries turn an unreliable text generator into something a program can depend on.
# - Offline self-tests with fake LLMs let me test the retry logic without spending API calls.
# - Always check the mistakes list, not only the average score: it shows *which kind* of ticket each prompt gets wrong.
