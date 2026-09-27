# Run the CPS / O*NET pipeline in order. Each target rebuilds only when its
# inputs change.  `make help` lists targets.

PY           ?= python
ONET_RELEASE ?= 30.3
ONET_DIR      = data/raw/onet_$(subst .,_,$(ONET_RELEASE))
OEWS_YEAR    ?= 2019

TASKS   = data/interim/task_composition.csv
OCC     = data/interim/occ_measures.csv
SAMPLE  = data/out/analysis_sample.parquet
RESULTS = data/out/results/rq1_task_composition.csv
PUBLIC  = data/raw/oews_national.xlsx data/raw/census_soc_crosswalk.xlsx data/raw/cpi_u.csv

.PHONY: help all fetch onet ipums usajobs usajobs-status estimate robustness test clean-interim

help:
	@echo "make fetch          00  OEWS, Census crosswalk, CPI-U (needs BLS_CONTACT_EMAIL)"
	@echo "make onet           01  O*NET $(ONET_RELEASE)"
	@echo "make ipums          04  IPUMS CPS extract (needs IPUMS_API_KEY)"
	@echo "make usajobs        06  USAJOBS historic announcements + panel"
	@echo "make estimate       02, 03, 05, 07 as needed"
	@echo "make robustness     07  with the pandemic window dropped"
	@echo "make test               unit and smoke tests"

all: estimate

fetch:
	$(PY) src/00_fetch_public_inputs.py --oews-year $(OEWS_YEAR)

onet: $(ONET_DIR)/_manifest.json
$(ONET_DIR)/_manifest.json:
	$(PY) src/01_download_onet.py --release $(ONET_RELEASE)

ipums:
	$(PY) src/04_ipums_extract.py

usajobs:
	$(PY) src/06_usajobs_historic.py

usajobs-status:
	$(PY) src/06_usajobs_historic.py --status

$(PUBLIC) data/raw/exposure_soc.csv:
	@echo "missing $@: run 'make fetch', or see data/README.md" >&2; exit 1

$(TASKS): src/02_build_task_composition.py mapping/onet_activity_map.csv $(ONET_DIR)/_manifest.json
	$(PY) src/02_build_task_composition.py --onet $(ONET_DIR) --out $@

$(OCC): src/03_crosswalk.py $(TASKS) $(PUBLIC) data/raw/exposure_soc.csv
	$(PY) src/03_crosswalk.py --out $@

$(SAMPLE): src/05_build_sample.py $(OCC) data/raw/cpi_u.csv
	$(PY) src/05_build_sample.py --out $@

$(RESULTS): src/07_estimate.py $(SAMPLE)
	$(PY) src/07_estimate.py --sample $(SAMPLE) --out data/out/results

estimate: $(RESULTS)

robustness: $(SAMPLE)
	$(PY) src/07_estimate.py --sample $(SAMPLE) --drop-pandemic --out data/out/results_drop_pandemic

test:
	$(PY) -m pytest -q

clean-interim:
	rm -f data/interim/*.csv data/out/*.parquet
