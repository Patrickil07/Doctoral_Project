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
| `raw/onet_27_0/` | O\*NET Resource Center, database release 27.0 (August 2022, the last before ChatGPT): **main task measures** | `src/01_download_onet.py --release 27.0` | release + SHA-256 from `_manifest.json` |
| `raw/onet_30_3/` | O\*NET release 30.3 (2025): GWA-name reference for the mapping and the `results_tasks_onet30` robustness run | `make onet` / `src/01_download_onet.py` | release + SHA-256 from `_manifest.json` |
| `raw/oews_national.xlsx` | BLS OEWS national estimates, **May 2021** | `make fetch` / `src/00_fetch_public_inputs.py` | year, URL, SHA-256 from `raw/_public_inputs_manifest.json` |
| `raw/census_soc_crosswalk.xlsx` | U.S. Census Bureau, 2018 Census Occupation Code List with Crosswalk (26 Sep 2019) | `make fetch` | URL, SHA-256 |
| `raw/soc_2010_to_2018_crosswalk.xlsx` | BLS SOC 2010 → 2018 crosswalk | `make fetch` | URL, SHA-256 |
| `raw/cpi_u.csv` | BLS CPI-U, all items, U.S. city average, NSA (`CUUR0000SA0`) → `year,month,cpi` | `make fetch` | series id, download date |
| `raw/eloundou_occ_level.csv` | **Primary exposure measure.** Eloundou, Manning, Mishkin & Rock (2024), *GPTs are GPTs*, `data/occ_level.csv` from github.com/openai/GPTs-are-GPTs at commit `0471612`; 923 O\*NET-SOC 2019 occupations | `make fetch` (pinned commit, SHA-256 `40c74f53…` checked) | paper, commit, SHA-256 |
| `raw/lm_aioe.xlsx` | Robustness measure: Felten, Raj & Seamans (2023), Language-Modeling AIOE, 774 occupations on **SOC 2010** codes | `make fetch`: `Language Modeling AIOE and AIIE.xlsx` from github.com/AIOE-Data/AIOE at commit `adca5fc` (SHA-256 `ccdd1fb9…` checked; identical to the copy in Drive `07_Data/raw/`) | paper, commit, SHA-256 |
| `interim/exposure_soc2018.csv` | primary measure: Eloundou **human-rated β** (E1 + 0.5·E2) on 6-digit SOC 2018 (the main .00 occupation where it exists, otherwise O\*NET-SOC detail averaged) | `make exposure` / `src/00b_convert_exposure.py` | column used, aggregation rule |
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
- **Task and exposure coverage differ.** A CPS code can span several SOC codes,
  and not every SOC code has an exposure score. Step 03 averages the task shares
  over the SOC codes with task data and exposure over the SOC codes with an
  exposure score, so z1–z3 are the same whichever exposure file is used and the
  exposure robustness runs change exposure only.
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
- **Imputed earnings are excluded (from run 9).** IPUMS does not accept
  allocation flags as variable names (the 27 September 2026 probe of 30 names
  such as `QEARNWEE` failed for that reason), but it delivers them through the
  data-quality-flags option of each variable. Step 04 requests that option for
  EARNWEEK, EARNWEEK2 and UHRSWORKORG; step 05 drops records whose earnings
  were allocated, logs the allocated share by quarter, and
  `results_keep_allocated` keeps them for comparison.
- **Hours.** Weekly earnings mix pay rates with hours, so the main earnings
  sample is full-time workers (usual weekly hours UHRSWORKORG 35-99; "hours
  vary" is excluded). `results_hourly_all_hours` keeps everyone with reported
  hours and uses log real hourly earnings.
- **Top-codes.** Real weekly earnings are censored at one common real cap (the
  lowest 2019-dollar value of the $2,884.61 nominal top-code over the sample
  months), so the move from a fixed to a monthly top-code in 2023-24 does not
  shift the post period; `results_drop_topcoded` drops the capped records.
- **Post period.** `post` = 2022Q4 onwards in every equation (the event
  studies already counted 2022Q4 as post). ChatGPT was released on 30 November
  2022, so 2022Q4 is two-thirds pre-release; report it as a partial quarter.
- **RQ2 sample.** The early-career employment share uses all employed
  wage/salary records in every rotation group, weighted by WTFINL
  (`analysis_sample_employment.parquet`), not only the ORG earners.
- **Knowledge universe.** Step 05 selects the knowledge-intensive groups from
  every Census code in the crosswalk (`interim/census_occ_codes.csv`), and logs
  the records in codes that lack measures before dropping them.
- **Crosswalk.** A Census "X" code (e.g. 13-20XX) covers only the SOCs no other
  Census row names; group codes such as 25-1000 cover their whole group; SOCs
  without OEWS employment get their broad (else minor) group's unpublished
  remainder, never a median.
- **Task shares.** Each part's score is the mean importance of its GWAs (the
  old sum made the shares mostly a count of GWAs per part); the sum is kept as
  `results_tasks_sum`, and `results_mapping_alt` uses the alternative mapping
  in `mapping/`.

## Where the data lives

On Google Drive, `doctoral project/07_Data/` holds `raw/`, `interim/` and `out/`
(the Colab runner links it to `data/`). `07_Data/sources/` keeps the original
downloads exactly as obtained, for provenance.

## Rebuild from scratch

```bash
export BLS_CONTACT_EMAIL=you@example.com
export IPUMS_API_KEY=...            # never commit this
make fetch onet ipums               # acquisition
make exposure                       # step 00b (Eloundou human-rated beta)
make estimate                       # steps 02 → 03 → 05 → 07
make robustness
make robustness-exposure            # GPT-4-rated beta, LM-AIOE
```

## Required citation (IPUMS)

Taken from the extract codebooks (`cps_00002.xml`, `cps_00003.xml`, both
produced 27 September 2026). Cite it exactly as below in any publication or
report that uses the data:

> Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, J. Robert Warren,
> Daniel Backman, Etienne Breton, Grace Cooper, Julia A. Rivera Drew, Stephanie
> Richards, David Van Riper, and Kari C.W. Williams. IPUMS CPS: Version 13.0
> [dataset]. Minneapolis, MN: IPUMS, 2025. https://doi.org/10.18128/D030.V13.0

The IPUMS licence also asks that the title and full citation of the
dissertation be added to the IPUMS bibliography (http://bibliography.ipums.org/).

### Extract used

| Extract | Months | Variables | Status |
|---|---|---|---|
| #3 (`cps_00003`) | 71: January 2020 – December 2025 except October 2025; no ASEC | YEAR, MONTH, EARNWT, STATEFIP, AGE, SEX, EDUC, EMPSTAT, CLASSWKR, IND, OCC, EARNWEEK, EARNWEEK2 (+ IPUMS preselected identifiers and weights) | **used** (`07_Data/raw/ipums/`) |
| #2 (`cps_00002`) | same 71 months, same variables | same | duplicate of #3; not used |
| #1 (`cps_00001`) | 20 months only | 13 variables | incomplete; archived, not used |

Codebook facts the pipeline relies on: EARNWEEK2 "not in universe" is
999999.99 and EARNWEEK 9999.99 (step 05 drops both); EARNWEEK2 carries the
$2,884.61 top-code while it was fixed and monthly (dynamic) top-codes after
it: IPUMS notes that from April 2023 to March 2024 only month-in-sample 4
records were rounded and dynamically top-coded, and step 05 prints the DDI
description to its log so the exact rule is on record (step 05 applies one
common real cap, see Recorded decisions); EARNWT is the earner-study weight. Unrounded EARNWEEK exists only to March 2023, so step 05 uses EARNWEEK2, which covers all 71 months. ASECFLAG is 2 (March basic) in March samples; step 05 drops any ASECFLAG = 1 record as a safeguard.
