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
| `raw/oews_national.xlsx` | BLS OEWS national estimates, **May 2021** | `make fetch` / `src/00_fetch_public_inputs.py` | year, URL, SHA-256 from `raw/_public_inputs_manifest.json` |
| `raw/census_soc_crosswalk.xlsx` | U.S. Census Bureau, 2018 Census Occupation Code List with Crosswalk (26 Sep 2019) | `make fetch` | URL, SHA-256 |
| `raw/soc_2010_to_2018_crosswalk.xlsx` | BLS SOC 2010 → 2018 crosswalk | `make fetch` | URL, SHA-256 |
| `raw/cpi_u.csv` | BLS CPI-U, all items, U.S. city average, NSA (`CUUR0000SA0`) → `year,month,cpi` | `make fetch` | series id, download date |
| `raw/exposure_soc2010.csv` | Felten, Raj & Seamans (2023), **Language-Modeling AIOE**, 774 occupations on **SOC 2010** codes (`soc2010,exposure`) | exported from `Language Modeling AIOE and AIIE.xlsx`, sheet *LM AIOE* | paper, file version, download date |
| `interim/exposure_soc2018.csv` | the exposure measure mapped to SOC 2018 | `make exposure` / `src/00b_convert_exposure.py` | mapping rules (split = copy, merge = mean) and the merged codes it lists |
| `raw/ipums/` | IPUMS CPS basic monthly samples 2019-01 to 2025-12 (`.xml` DDI + `.dat.gz`) | `make ipums` / `src/04_ipums_extract.py` (needs `IPUMS_API_KEY`, CPS registration) | extract number + IPUMS CPS version from the DDI |
| `raw/usajobs/` | USAJOBS historic JOA API (public, no key) | `make usajobs` / `src/06_usajobs_historic.py` | series list, date range, retrieval dates |

### Coding pitfalls found in the inputs

- **Exposure measures on SOC 2010.** AIOE and LM-AIOE use SOC 2010 codes. Merged
  directly on SOC 2018 they miss every occupation recoded in 2018, including
  all of 15-12xx (software developers, analysts, database and network roles).
  Always go through step 00b. Eloundou et al. (2024), which the proposal names
  as the primary measure, is published on O\*NET-SOC 2019 (SOC 2018 based):
  use `--already-2018`.
- **OEWS 2019/2020 hybrid codes.** Those years publish some occupations under
  combined codes (e.g. 15-1256 for 15-1252 + 15-1253), so detailed occupations
  get no employment weight. May 2021 is the first fully SOC 2018 release and
  is still before the treatment date.
- **CPS occupation codes before 2020.** CPS records before January 2020 carry
  2010 Census occupation codes; the occupation measures are keyed on 2018
  Census codes. Step 05 warns about this. Either recode 2019 records with the
  *2010 to 2018 Crosswalk* sheet of `census_soc_crosswalk.xlsx`, or restrict to
  the `census2018` regime (proposal Section 6.6). This choice is open.

## Where the data lives

On Google Drive, `doctoral project/07_Data/` holds `raw/`, `interim/` and `out/`
(the Colab runner links it to `data/`). `07_Data/sources/` keeps the original
downloads exactly as obtained, for provenance.

## Rebuild from scratch

```bash
export BLS_CONTACT_EMAIL=you@example.com
export IPUMS_API_KEY=...            # never commit this
make fetch onet ipums usajobs       # acquisition
# place data/raw/exposure_soc2010.csv (see table above)
make exposure                       # step 00b
make estimate                       # steps 02 → 03 → 05 → 07
make robustness
```

## Required citation (IPUMS)

Cite IPUMS CPS exactly as IPUMS specifies on the extract's download page
(authors, version number, DOI). Take the version from the extract's DDI
codebook, not from memory.
