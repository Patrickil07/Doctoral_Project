# The Pipeline Paradox

Analysis code for the DBA dissertation *The Pipeline Paradox*: whether
generative-AI exposure changes the task composition, employment share and
wage returns of early-career knowledge workers (RQ1–RQ4).

This repository holds **code only**. Chapter drafts, supervisor feedback,
ethics paperwork and reading notes live in Google Drive. Data is never
committed (see [`data/README.md`](data/README.md)).

Google Drive layout (`My Drive/doctoral project/`):

```
01_Proposal/          research proposal versions
02_Dissertation/      the dissertation draft (chapters)
03_Literature/        papers, O*NET technical documentation, reading notes
04_Supervision/       supervisor feedback, meeting notes
05_Ethics_and_Admin/  ethics approval, forms, milestones
06_Presentations/     slides
07_Data/              raw/ interim/ out/ + sources/ (linked to data/ on Colab)
99_Archive/           superseded drafts and old code copies (code now lives here on GitHub)
```

## Layout

```
.
├── src/                         the pipeline, one script per step, run in order
│   ├── 00_fetch_public_inputs.py    OEWS, Census + BLS SOC crosswalks, CPI-U
│   ├── 00b_convert_exposure.py      exposure measure (Eloundou et al. 2024) → 6-digit SOC 2018
│   ├── 01_download_onet.py          pinned O*NET release
│   ├── 02_build_task_composition.py four-part task composition + ILR balances
│   ├── 03_crosswalk.py              SOC → CPS occupation codes, OEWS weights, exposure
│   ├── 04_ipums_extract.py          IPUMS CPS extract via the IPUMS API
│   ├── 05_build_sample.py           estimation sample (Table 4.1 log)
│   ├── 06_figures_tables.py         Chapter 4 figures + formatted tables, from saved results only
│   ├── 07_estimate.py               RQ1–RQ4 (Eqs. 5–7) and the RQ1 specification menu
│   └── 08_honest_did.R              Rambachan & Roth (2023) sensitivity for RQ1 (HonestDiD R package)
├── mapping/                     O*NET activity → task-part mapping (a design choice; versioned)
├── notebooks/
│   └── run_pipeline_colab.ipynb     runs src/ on Colab: code from GitHub, data on Drive
├── results/                     aggregate tables cited in chapters (committed + tagged)
├── tests/                       unit and smoke tests (made-up inputs, never results)
├── docs/results_provenance.md   how chapter tables are tied to tagged commits
├── data/                        git-ignored; rebuilt from src/ (see data/README.md)
├── Makefile                     `make help`
└── requirements*.txt
```

Step 06 (figures and tables) runs after step 07 because it only reads the
result files 07 writes; the number was freed when the USAJOBS strand was removed.

Each `notebooks/*.ipynb` has a `.py` twin (jupytext) so changes are readable in diffs.

## Getting started

```bash
git clone https://github.com/patrickil07/doctoral_project.git
cd doctoral_project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
nbstripout --install           # once per clone: strips notebook outputs on commit
make test
```

On Colab, open `notebooks/run_pipeline_colab.ipynb` and follow the secrets
table at the top.

## Running the analysis (automated, recommended)

The whole pipeline runs on GitHub's machines with one click; no Colab needed.

1. Once: add the repository secret `IPUMS_API_KEY` (Settings → Secrets and
   variables → Actions → New repository secret).
2. Actions tab → **run-pipeline** → **Run workflow** (extract number defaults to 3;
   an expired extract is re-submitted automatically with the same definition).
3. When it finishes, download **pipeline-results** from the run page: result
   tables, occupation measures, the Table 4.1 sample log (`logs/05_sample.log`)
   and `logs/provenance.txt` (commit, run URL, extract number, input checksums).
4. The same files are saved permanently on the `results` branch, one folder
   per run: `runs/<date>_run<N>_<commit>/`. The run artifact expires after 90
   days; the branch copy does not. Each run's folder is also copied to Google
   Drive, `07_Data/out/pipeline_runs/<run>/`, once the Drive token below is set.

Every input is downloaded fresh; IPUMS microdata stays on the runner and is
deleted afterwards. Only aggregate outputs are kept.

### Google Drive copy

Every run folder (aggregate results, logs, `chapter4/`; never microdata) is
copied to `07_Data/out/pipeline_runs/<run>/` in Google Drive by
`.github/sync_drive.sh`, and each file is checked against the original (size
and MD5). run-pipeline copies each new run, build-chapter4 re-copies a run
after rebuilding it, and **sync-drive** copies saved runs on demand (a run
name, empty for the newest, or `all`). A Drive problem never fails a run.

One-time setup (on your own computer, not Colab, because it opens a browser):

1. Install rclone (<https://rclone.org/downloads/>).
2. Run `rclone authorize "drive"`, sign in with the Google account that owns
   the Drive folder and allow access. rclone prints a token: the text starting
   with `{"access_token"` and ending with `}`.
3. Add it as the repository secret `RCLONE_DRIVE_TOKEN` (Settings → Secrets and
   variables → Actions → New repository secret).
4. Actions tab → **sync-drive** → **Run workflow** with `all` to copy the runs
   saved so far.

The target is the `pipeline_runs` folder; to use another one, set the
repository variable `GDRIVE_FOLDER_ID` to that folder's id (the last part of
its Drive URL).

### RQ1 specification menu

Every run estimates RQ1 three ways, each saved as its own results folder:

| Model | Folder | What it does |
|---|---|---|
| 1 Baseline | `results/` | Event study over 2020–2025 (Eq. 5), pre-trend shown as estimated |
| 2 Linear trend | `results_rq1_trend/` | Adds E×J×t (t = quarters from 2022Q3) and keeps post-quarter dummies only, so the trend is fitted on the pre-period and each post coefficient is the deviation from its extrapolation (Dobkin et al. 2018) |
| 3 From 2021Q4 | `results_rq1_from_2021q4/` | Same as Model 1 on 2021Q4–2025Q4 |

With every pre-period dummy kept, a linear E×J×t term would be perfectly
collinear with them, which is why Model 2 drops them. Each event study also
saves its clustered covariance matrix (`*_vcov.csv`), which gives the joint
pre-trend Wald test and feeds step 08: the Rambachan & Roth (2023)
relative-magnitudes sensitivity for the average post-period RQ1 coefficient
(Models 1 and 3; Model 2 has no pre-period coefficients), including the
breakdown value M̄.

### Chapter 4 figures and tables (step 06)

Each pipeline run ends with step 06, which turns the saved result files into
figures and formatted tables in `chapter4/` next to the results:
`figures/` (PNG at 300 dpi and PDF), `tables/` (CSV, Markdown, LaTeX) and
`chapter4_tables.xlsx` (every table on its own sheet, for pasting into Word).
It never touches microdata.

To rebuild them for a run that is already saved, without re-running the
pipeline (e.g. after changing a figure): Actions tab → **build-chapter4** →
**Run workflow** (leave the run folder empty for the newest run). Locally:
`make figures RUN=<path to a run folder copied from the results branch>`.

Every `data/out/results*/` folder is one specification and shows up in the
tables and comparison figures automatically. To add one (for example an extra
RQ1 specification), write it with `07_estimate.py --out data/out/results_<name>`
and give it a readable name in `SPEC_LABELS` in `src/06_figures_tables.py`.

## Running the pipeline locally or on Colab

```bash
export BLS_CONTACT_EMAIL=you@example.com
export IPUMS_API_KEY=...        # never commit; on Colab use the Secrets panel
make fetch onet ipums           # data acquisition
make exposure                   # Eloundou human-rated beta → SOC 2018
make estimate                   # 02 → 03 → 05 → 07, rebuilding only what changed
make robustness                 # pandemic window dropped
make robustness-exposure        # GPT-4-rated beta, LM-AIOE
make rq1-models                 # RQ1 Model 2 (linear trend), Model 3 (from 2021Q4)
make honest-did                 # 08: Rambachan & Roth sensitivity (needs R + HonestDiD)
make figures                    # 06: Chapter 4 figures + tables -> data/out/chapter4/
```

Outputs land in `data/out/results/`. To cite results in a chapter, follow
[`docs/results_provenance.md`](docs/results_provenance.md).

## Working rules

- **One change per commit, with a message saying what and why.** Commit
  history is the dated audit trail of the empirical work.
- **Never commit** data, API keys, notebook outputs, supervision notes or
  drafts. `.gitignore` and `nbstripout` enforce most of this; the rest is on
  you. Git history is hard to erase.
- **Synthetic data is for testing code only.** `tests/` exists to check that
  code runs. Nothing produced from it is a result.
- **Design choices live in versioned files** (`mapping/`, script arguments),
  not in ad-hoc edits, so each one can be defended with its commit.

## Data sources

O\*NET (U.S. Department of Labor), BLS OEWS and CPI-U, U.S. Census Bureau
occupation crosswalk and BLS SOC 2010→2018 crosswalk, IPUMS CPS (University of
Minnesota), Eloundou et al. (2024) and Felten, Raj & Seamans (2023) exposure
measures. Full provenance and citation requirements: [`data/README.md`](data/README.md).
