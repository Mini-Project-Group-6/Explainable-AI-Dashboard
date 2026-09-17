# Explainable AI Co-Teaching Dashboard

A dashboard for Ghana's Colleges of Education. A student teacher uploads a
lesson plan; the dashboard scores it against ten criteria from the National
Teachers' Standards and shows *why* each score was given (SHAP explanations),
with suggested revisions. It tracks scores across drafts and runs a short
before-and-after survey on how far student teachers trust the feedback.

> The models were trained on **synthetic** lesson plans. The scores show how
> the system works; they have not been validated on real plans.

## What you can do in it

| Tab | What it shows |
|---|---|
| Scoring | Upload a plan (PDF, Word or .txt). You get the overall score, the ten criterion scores with their evidence, SHAP waterfall and force plots, and revision suggestions |
| Revision history | Every scored draft, with a trend chart |
| Trust survey | Consent, the first and follow-up surveys, and withdrawal. Researcher accounts see enrolment counts and an anonymised CSV export |
| Transparency | What the model measures and what it cannot see. Researcher accounts also see deployment details |

## Running it

Tested on **Windows 11** with **Python 3.12** and **PostgreSQL 18** (17 also
works). The steps below are for Windows PowerShell; on macOS or Linux use
`python3.12` and `.venv/bin/python` instead. Those systems have not been
tested.

You need:
- [Python 3.12](https://www.python.org/downloads/). Other versions do not
  work with the pinned packages.
- [PostgreSQL](https://www.postgresql.org/download/). Note the password you
  set for the `postgres` user during installation.
- Internet for the first install and the first run. Roughly 1–1.5 GB is
  downloaded in total, including the DistilBERT base model on first start.

### 1. Get the code

```powershell
git clone https://github.com/Mini-Project-Group-6/Explainable-AI-Dashboard.git
cd Explainable-AI-Dashboard
```

The trained models (about 20 MB) and the sample lesson plans are in the
repository, so nothing needs training.

### 2. Install the Python packages

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r model/requirements.txt
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m spacy download en_core_web_sm
.venv\Scripts\python -m pip install -r requirements.txt
```

Install PyTorch from the CPU index as shown. The default Windows package
includes CUDA and is about 2.4 GB.

### 3. Create the database

```powershell
.venv\Scripts\python -m app.database setup
.venv\Scripts\python -m app.database check
```

`setup` asks once for the `postgres` password. It then:
- finds your running PostgreSQL server (on Windows it reads the port from the
  installer, so a server on 5433 is found too);
- creates a `coteach` login and database;
- writes the connection details, with a generated password, to `.env`.

`check` should print the server version and four empty tables. If
PostgreSQL is not on this machine, or is on another port, pass
`--host`/`--port` to `setup`.

<details>
<summary>No PostgreSQL install? Use Docker instead</summary>

```powershell
docker compose up -d
copy .env.example .env
```

Then edit `.env`: set `DB_PORT=5434` and `DB_PASSWORD=coteach_dev_only`.
</details>

### 4. Create accounts

There is no sign-up page, so create accounts from the terminal. The addresses
do not need to be real. `--generate` prints a password once, so copy it.

```powershell
.venv\Scripts\python -m app.accounts add student@example.com --name "Demo Student" --generate
.venv\Scripts\python -m app.accounts add researcher@example.com --name "Demo Researcher" --role researcher --generate
```

A whole class can be created from a CSV file with
`python -m app.accounts import class.csv --passwords <new file>`.

### 5. Start the dashboard

```powershell
.venv\Scripts\python -m streamlit run main.py
```

Open <http://localhost:8501>. The first page load takes about 15 seconds
while the models load. The very first start also downloads DistilBERT.

### 6. A walk-through

1. **Sign in as the student.** A *Research participation* panel appears. It
   is marked **pilot version**, because ethics approval is still pending.
   - Choose **I agree** to see the whole study flow.
   - Choose **No thanks** to go straight to scoring.
2. **Trust survey tab** (if you agreed): answer the first survey. Uploads stay
   locked until you do, so that the "before" answers come before any AI
   feedback.
3. **Scoring tab:** upload `model/data/synthetic_revisions/plans/rev_0002_v1.txt`.
   You get scores, evidence, SHAP plots and suggestions.
4. **Upload the revised draft,** `rev_0002_v2.txt`. The **Revision history**
   tab shows the score rising (38.7 → 57.0), and the follow-up survey opens.
5. **Sign in as the researcher.** The Trust survey tab shows enrolment and the
   CSV export. The Transparency tab shows deployment details.

More sample plans are in `model/data/synthetic_v1/plans/` (200 plans) and
`model/data/synthetic_revisions/plans/` (80 first drafts and their
revisions).

### Tests

```powershell
.venv\Scripts\python -m unittest discover -s tests -t .
cd model; ..\.venv\Scripts\python -m unittest discover -s tests
```

The dashboard tests use a throwaway SQLite file, so they never touch your
database. To run them against PostgreSQL instead, set
`COTEACH_TEST_DATABASE_URL` to a database whose name ends in `_test`.

## Layout

| Path | Contents | Owner |
|---|---|---|
| `model/` | Feature extraction, XGBoost and DistilBERT scoring, SHAP explanations and their UI components | S1, S2 |
| `main.py`, `pages/`, `app/` | Sign-in, dashboard, storage, accounts, database tools | S3 |
| `app/study/`, `docs/irb/` | Survey instruments, consent, data collection, ethics drafts | S4 |
| `tests/` | Dashboard, storage, account and study tests | S3, S4 |
| `DEV.md` | Design notes, measured results and conventions | all |

## Tech stack

Python 3.12 · Streamlit · PostgreSQL (SQLAlchemy) · XGBoost · DistilBERT with
LoRA (PyTorch, Transformers, PEFT) · SHAP · spaCy
