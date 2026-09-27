# Data: what goes here and how to rebuild it

Nothing in `data/` is committed except this file and the empty folder markers.
IPUMS terms of use prohibit redistributing extracts, and every file below can be
rebuilt from the scripts in `src/`. On Colab, `data/` is a link to the Drive
folder so large downloads persist between sessions.

```
data/
├── raw/        downloaded inputs, never edited by hand (except exposure_soc.csv)
├── interim/    derived occupation-level measures (steps 02, 03, 06 panel)
└── out/        estimation sample and results (steps 05, 07)
```

## Inputs

| File | Source | How it is obtained | Record in methodology |
|---|---|---|---|
| `raw/onet_30_3/` | O\*NET Resource Center, database release 30.3 (text files) | `make onet` / `src/01_download_onet.py` | release + SHA-256 from `_manifest.json` |
| `raw/oews_national.xlsx` | BLS Occupational Employment and Wage Statistics, national, May 2019 | `make fetch` / `src/00_fetch_public_inputs.py` | year, URL, SHA-256 from `raw/_public_inputs_manifest.json` |
| `raw/census_soc_crosswalk.xlsx` | U.S. Census Bureau, 2018 Census Occupation Code List with Crosswalk | `make fetch` | URL, SHA-256 |
| `raw/cpi_u.csv` | BLS CPI-U, all items, U.S. city average, NSA (`CUUR0000SA0`) → `year,month,cpi` | `make fetch` | series id, download date |
| `raw/exposure_soc.csv` | **To document:** GenAI occupational exposure measure, columns `soc2018,exposure` | built by hand | source, version, any SOC recoding |
| `raw/ipums/` | IPUMS CPS basic monthly samples 2019-01 to 2025-12 (`.xml` DDI + `.dat.gz`) | `make ipums` / `src/04_ipums_extract.py` (needs `IPUMS_API_KEY`, CPS registration) | extract number + IPUMS CPS version from the DDI |
| `raw/usajobs/` | USAJOBS historic JOA API (public, no key) | `make usajobs` / `src/06_usajobs_historic.py` | series list, date range, retrieval dates |

## Rebuild from scratch

```bash
export BLS_CONTACT_EMAIL=you@example.com
export IPUMS_API_KEY=...            # never commit this
make fetch onet ipums usajobs       # acquisition
# place data/raw/exposure_soc.csv
make estimate                       # steps 02 → 03 → 05 → 07
make robustness
```

## Required citation (IPUMS)

Cite IPUMS CPS exactly as IPUMS specifies on the extract's download page
(authors, version number, DOI). Take the version from the extract's DDI
codebook, not from memory.
