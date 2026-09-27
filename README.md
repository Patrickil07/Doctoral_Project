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
│   ├── 06_usajobs_historic.py       USAJOBS historic announcements (federal strand)
│   └── 07_estimate.py               RQ1–RQ4 (Eqs. 5–7)
├── mapping/                     O*NET activity → task-part mapping (a design choice; versioned)
├── notebooks/
│   ├── run_pipeline_colab.ipynb     runs src/ on Colab: code from GitHub, data on Drive
│   └── vacancy_text_lda.ipynb       vacancy-text LDA strand (Indeed 2021 / LinkedIn 2024)
├── results/                     aggregate tables cited in chapters (committed + tagged)
├── tests/                       unit and smoke tests (made-up inputs, never results)
├── docs/results_provenance.md   how chapter tables are tied to tagged commits
├── data/                        git-ignored; rebuilt from src/ (see data/README.md)
├── Makefile                     `make help`
└── requirements*.txt
```

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

## Running the pipeline

```bash
export BLS_CONTACT_EMAIL=you@example.com
export IPUMS_API_KEY=...        # never commit; on Colab use the Secrets panel
make fetch onet ipums usajobs   # data acquisition
make exposure                   # Eloundou human-rated beta → SOC 2018
make estimate                   # 02 → 03 → 05 → 07, rebuilding only what changed
make robustness                 # pandemic window dropped
make robustness-exposure        # GPT-4-rated beta, LM-AIOE
```

Outputs land in `data/out/results/`. To cite results in a chapter, follow
[`docs/results_provenance.md`](docs/results_provenance.md).

## Working rules

- **One change per commit, with a message saying what and why.** Commit
  history is the dated audit trail of the empirical work.
- **Never commit** data, API keys, notebook outputs, supervision notes or
  drafts. `.gitignore` and `nbstripout` enforce most of this; the rest is on
  you. Git history is hard to erase.
- **Synthetic data is for testing code only.** `tests/` and the LDA notebook's
  `USE_SYNTHETIC_DATA` switch exist to check that code runs. Nothing produced
  from them is a result.
- **Design choices live in versioned files** (`mapping/`, script arguments),
  not in ad-hoc edits, so each one can be defended with its commit.

## Data sources

O\*NET (U.S. Department of Labor), BLS OEWS and CPI-U, U.S. Census Bureau
occupation crosswalk, IPUMS CPS (University of Minnesota), USAJOBS historic JOA
API. Full provenance and citation requirements: [`data/README.md`](data/README.md).
