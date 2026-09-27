# Data: what goes here and how to rebuild it

Nothing in `data/` is committed except this file and the empty folder markers.
IPUMS terms of use prohibit redistributing extracts, and every file below can be
rebuilt from the scripts in `src/`. On Colab, `data/` is a link to the Drive
folder so large downloads persist between sessions.

```
data/
├── raw/        downloaded inputs, never edited by hand
├── interim/    derived occupation-level measures (steps 00b, 02, 03)
└── out/        estimation sample and results (steps 05, 07)
```

## Inputs

| File | Source | How it is obtained | Record in methodology |
|---|---|---|---|
| `raw/onet_30_3/` | O\*NET Resource Center, database release 30.3 (text files) | `make onet` / `src/01_download_onet.py` | release + SHA-256 from `_manifest.json` |
| `raw/oews_national.xlsx` | BLS OEWS national estimates, **May 2021** | `make fetch` / `src/00_fetch_public_inputs.py` | year, URL, SHA-256 from `raw/_public_inputs_manifest.json` |
| `raw/census_soc_crosswalk.xlsx` | U.S. Census Bureau, 2018 Census Occupation Code List with Crosswalk (26 Sep 2019) | `make fetch` | URL, SHA-256 |
| `raw/soc_2010_to_2018_crosswalk.xlsx` | BLS SOC 2010 → 2018 crosswalk | `make fetch` | URL, SHA-256 |
| `raw/cpi_u.csv` | BLS CPI-U, all items, U.S. city average, NSA (`CUUR0000SA0`) → `year,month,cpi` | `make fetch` | series id, download date |
| `raw/eloundou_occ_level.csv` | **Primary exposure measure.** Eloundou, Manning, Mishkin & Rock (2024), *GPTs are GPTs*, `data/occ_level.csv` from github.com/openai/GPTs-are-GPTs at commit `0471612`; 923 O\*NET-SOC 2019 occupations | `make fetch` (pinned commit, SHA-256 `40c74f53…` checked) | paper, commit, SHA-256 |
| `raw/lm_aioe.xlsx` | Robustness measure: Felten, Raj & Seamans (2023), Language-Modeling AIOE, 774 occupations on **SOC 2010** codes | copy of `Language Modeling AIOE and AIIE.xlsx` (sheet *LM AIOE*); original in `07_Data/sources/` | paper, file version, download date |
| `interim/exposure_soc2018.csv` | primary measure: Eloundou **human-rated β** (E1 + 0.5·E2) on 6-digit SOC 2018 (O\*NET-SOC detail averaged) | `make exposure` / `src/00b_convert_exposure.py` | column used, aggregation rule |
| `interim/exposure_{gpt4beta,lmaioe}_soc2018.csv` | robustness measures: Eloundou GPT-4-rated β; LM-AIOE mapped from SOC 2010 (split = copy, merge = mean) | `make robustness-exposure` | as above, plus the merged codes step 00b lists |
| `raw/ipums/` | IPUMS CPS monthly samples 2020-01 to 2025-12, **71 months** (October 2025 is not offered by IPUMS); months that carried a supplement are IPUMS `…s` samples, the March ASEC is excluded (`.xml` DDI + `.dat.gz`, plus `_extract_request.json`) | `make ipums` / `src/04_ipums_extract.py` (needs `IPUMS_API_KEY`, CPS registration) | extract number + IPUMS CPS version from the DDI |

### Coding pitfalls found in the inputs

- **Exposure measures on SOC 2010.** AIOE and LM-AIOE use SOC 2010 codes. Merged
  directly on SOC 2018 they miss every occupation recoded in 2018, including
  all of 15-12xx (software developers, analysts, database and network roles).
  Step 00b converts them (`--source-soc 2010`). The primary measure, Eloundou
  et al. (2024), is on O\*NET-SOC 2019 (SOC 2018 based) and needs no conversion;
  it covers 180 of the 189 CPS occupation codes in the knowledge-intensive
  groups, missing only 'All Other' residual codes that O\*NET also leaves unrated.
- **Exposure scales differ.** Eloundou scores are shares of tasks in [0, 1];
  AIOE scores are standardised. Coefficients on exposure are not comparable
  across the primary and LM-AIOE results without rescaling.
- **OEWS 2019/2020 hybrid codes.** Those years publish some occupations under
  combined codes (e.g. 15-1256 for 15-1252 + 15-1253), so detailed occupations
  get no employment weight. May 2021 is the first fully SOC 2018 release and
  is still before the treatment date.
- **CPS occupation codes before 2020.** CPS records before January 2020 carry
  2010 Census occupation codes; the occupation measures are keyed on 2018
  Census codes. **Decision: the sample starts in January 2020.** Step 04
  requests 2020-01 onwards and step 05 enforces it (an earlier `--start` is
  refused). Consequences to report: the pre-treatment window is 2020Q1-2022Q2
  (10 quarters, reference 2022Q3), and it overlaps the pandemic window
  (2020Q2-2021Q2); with `--drop-pandemic` five pre-treatment quarters remain
  (2020Q1, 2021Q3-2022Q2). Earnings are still expressed in 2019 dollars.

### Recorded decisions

- **October 2025 is excluded.** IPUMS offers no CPS monthly sample for that
  month, and BLS published no CPI-U for it (blank in `raw/cpi_u.csv`). The
  sample therefore covers 71 months; no value is interpolated.
- **SOC major group from the Census crosswalk.** IPUMS CPS has no `OCCSOC`
  variable; step 03 assigns each CPS occupation code its SOC 2018 major group
  (largest-employment group where a code spans several) and step 05 selects
  the knowledge-intensive groups with it.
- **Imputed earnings: open.** The extract has no EARNWEEK allocation flag
  (`QEARNWEEK` and `QEARNWEE` are not IPUMS CPS names). Until the correct flag
  is added, step 05 keeps imputed earnings and prints a warning; the planned
  Hirsch-Schumacher exclusion is not applied.

## Where the data lives

On Google Drive, `doctoral project/07_Data/` holds `raw/`, `interim/` and `out/`
(the Colab runner links it to `data/`). `07_Data/sources/` keeps the original
downloads exactly as obtained, for provenance.

## Rebuild from scratch

```bash
export BLS_CONTACT_EMAIL=you@example.com
export IPUMS_API_KEY=...            # never commit this
make fetch onet ipums               # acquisition
# place data/raw/lm_aioe.xlsx (see table above)
make exposure                       # step 00b (Eloundou human-rated beta)
make estimate                       # steps 02 → 03 → 05 → 07
make robustness
make robustness-exposure            # GPT-4-rated beta, LM-AIOE
```

## Required citation (IPUMS)

Cite IPUMS CPS exactly as IPUMS specifies on the extract's download page
(authors, version number, DOI). Take the version from the extract's DDI
codebook, not from memory.
