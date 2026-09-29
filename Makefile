# Run the CPS / O*NET pipeline in order. Each target rebuilds only when its
# inputs change.  `make help` lists targets.

PY           ?= python
ONET_RELEASE ?= 30.3
ONET_DIR      = data/raw/onet_$(subst .,_,$(ONET_RELEASE))
OEWS_YEAR    ?= 2021

TASKS   = data/interim/task_composition.csv
OCC     = data/interim/occ_measures.csv
SAMPLE  = data/out/analysis_sample.parquet
RESULTS = data/out/results/rq1_task_composition.csv
PUBLIC  = data/raw/oews_national.xlsx data/raw/census_soc_crosswalk.xlsx data/raw/cpi_u.csv
EXPOSURE = data/interim/exposure_soc2018.csv

.PHONY: help all fetch exposure robustness-exposure onet ipums estimate robustness rq1-models honest-did figures test clean-interim

help:
	@echo "make fetch          00  OEWS, Census + SOC crosswalks, CPI-U (needs BLS_CONTACT_EMAIL)"
	@echo "make exposure       00b Eloundou et al. human-rated beta -> SOC 2018 (primary)"
	@echo "make onet           01  O*NET $(ONET_RELEASE)"
	@echo "make ipums          04  IPUMS CPS extract (needs IPUMS_API_KEY)"
	@echo "make estimate       02, 03, 05, 07 as needed"
	@echo "make robustness     07  with the pandemic window dropped"
	@echo "make robustness-exposure  03-07 with GPT-4-rated beta and with LM-AIOE"
	@echo "make rq1-models     07  RQ1 Model 2 (linear trend) and Model 3 (window from 2021Q4)"
	@echo "make honest-did     08  Rambachan & Roth sensitivity for RQ1 (needs R + HonestDiD)"
	@echo "make figures        06  Chapter 4 figures + tables from saved results (RUN=runs/<run> for a saved run)"
	@echo "make test               unit and smoke tests"

all: estimate

fetch:
	$(PY) src/00_fetch_public_inputs.py --oews-year $(OEWS_YEAR)

onet: $(ONET_DIR)/_manifest.json
$(ONET_DIR)/_manifest.json:
	$(PY) src/01_download_onet.py --release $(ONET_RELEASE)

ipums:
	$(PY) src/04_ipums_extract.py

exposure: $(EXPOSURE)
$(EXPOSURE): src/00b_convert_exposure.py data/raw/eloundou_occ_level.csv
	$(PY) src/00b_convert_exposure.py --out $@

$(PUBLIC) data/raw/eloundou_occ_level.csv data/raw/lm_aioe.xlsx data/raw/soc_2010_to_2018_crosswalk.xlsx:
	@echo "missing $@: run 'make fetch', or see data/README.md" >&2; exit 1

$(TASKS): src/02_build_task_composition.py mapping/onet_activity_map.csv $(ONET_DIR)/_manifest.json
	$(PY) src/02_build_task_composition.py --onet $(ONET_DIR) --out $@

$(OCC): src/03_crosswalk.py $(TASKS) $(PUBLIC) $(EXPOSURE)
	$(PY) src/03_crosswalk.py --out $@

$(SAMPLE): src/05_build_sample.py $(OCC) data/raw/cpi_u.csv
	$(PY) src/05_build_sample.py --out $@

$(RESULTS): src/07_estimate.py $(SAMPLE)
	$(PY) src/07_estimate.py --sample $(SAMPLE) --out data/out/results

estimate: $(RESULTS)

robustness: $(SAMPLE)
	$(PY) src/07_estimate.py --sample $(SAMPLE) --drop-pandemic --out data/out/results_drop_pandemic

# Alternative exposure measures, run through the same 03 -> 05 -> 07 chain.
robustness-exposure: $(TASKS) data/raw/eloundou_occ_level.csv data/raw/lm_aioe.xlsx data/raw/soc_2010_to_2018_crosswalk.xlsx
	$(PY) src/00b_convert_exposure.py --column dv_rating_beta --out data/interim/exposure_gpt4beta_soc2018.csv
	$(PY) src/00b_convert_exposure.py --exposure data/raw/lm_aioe.xlsx --column "Language Modeling AIOE" --source-soc 2010 --out data/interim/exposure_lmaioe_soc2018.csv
	for m in gpt4beta lmaioe; do \
	  $(PY) src/03_crosswalk.py --exposure data/interim/exposure_$${m}_soc2018.csv --out data/interim/occ_measures_$$m.csv && \
	  $(PY) src/05_build_sample.py --occ data/interim/occ_measures_$$m.csv --out data/out/analysis_sample_$$m.parquet && \
	  $(PY) src/07_estimate.py --sample data/out/analysis_sample_$$m.parquet --out data/out/results_exposure_$$m || exit 1; \
	done

# RQ1 specification menu: Model 1 is the main run (make estimate).
rq1-models: $(SAMPLE)
	$(PY) src/07_estimate.py --sample $(SAMPLE) --rq1-trend --rq1-only --out data/out/results_rq1_trend
	$(PY) src/07_estimate.py --sample $(SAMPLE) --start-quarter 2021Q4 --rq1-only --out data/out/results_rq1_from_2021q4

honest-did:
	Rscript src/08_honest_did.R --results data/out/results
	Rscript src/08_honest_did.R --results data/out/results_rq1_from_2021q4

# Reads saved results only (every data/out/results*/ folder), never microdata.
RUN ?= .
figures:
	$(PY) src/06_figures_tables.py --root $(RUN) --out $(if $(filter .,$(RUN)),data/out/chapter4,$(RUN)/chapter4)

test:
	$(PY) -m pytest -q

clean-interim:
	rm -f data/interim/*.csv data/out/*.parquet
