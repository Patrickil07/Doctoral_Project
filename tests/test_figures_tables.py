"""
Smoke tests for src/06_figures_tables.py on made-up result files.

The inputs mimic what steps 05 and 07 write (same columns, invented numbers).
Nothing here is a result.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(stem: str):
    spec = importlib.util.spec_from_file_location(stem, ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fig = load("06_figures_tables")

SAMPLE_LOG = """[sample] reading cps_00003.dat in chunks of 500,000

[sample] SAMPLE CONSTRUCTION LOG (paste into Table 4.1):
    raw records                                       1,000
    in ORG earner universe                              400
    valid weekly earnings                               390

[sample] early-career: 90 | experienced: 300
"""


def _event(prefix: str, quarters: list, rng) -> pd.DataFrame:
    qs = [q for q in quarters if q != fig.REF_Q]
    return pd.DataFrame({"term": [f"{prefix}_{q}" for q in qs],
                         "estimate": rng.normal(0, .05, len(qs)),
                         "se": rng.uniform(.01, .03, len(qs)),
                         "pvalue": rng.uniform(0, 1, len(qs)), "quarter": qs,
                         "period": ["pre" if q < fig.REF_Q else "post" for q in qs]})


def _write_spec(folder: pathlib.Path, rng, drop: tuple = ()):
    quarters = [q for q in pd.period_range("2021Q1", "2024Q4", freq="Q").astype(str)
                if q not in drop]
    folder.mkdir(parents=True)
    for f in ("rq1_task_composition", "rq4_earnings_unconditional", "rq4_earnings_conditional"):
        _event("expJq", quarters, rng).to_csv(folder / f"{f}.csv", index=False)
    _event("expq", quarters, rng).to_csv(folder / "rq2_early_share.csv", index=False)
    u = pd.read_csv(folder / "rq4_earnings_unconditional.csv")
    c = pd.read_csv(folder / "rq4_earnings_conditional.csv")
    m = u.merge(c, on="term", suffixes=("_uncond", "_cond"))
    m["composition_component"] = m["estimate_uncond"] - m["estimate_cond"]
    m.to_csv(folder / "rq4_decomposition.csv", index=False)
    terms = list(fig.RQ3_TERMS)
    pd.DataFrame({"term": terms, "estimate": rng.normal(0, .2, len(terms)),
                  "se": rng.uniform(.05, .1, len(terms)),
                  "pvalue": rng.uniform(0, 1, len(terms))}).to_csv(folder / "rq3_hedonic.csv",
                                                                  index=False)


@pytest.fixture
def run_root(tmp_path):
    rng = np.random.default_rng(3)
    root = tmp_path / "run"
    _write_spec(root / "data" / "out" / "results", rng)
    _write_spec(root / "data" / "out" / "results_drop_pandemic", rng, drop=("2021Q1", "2021Q2"))
    _write_spec(root / "data" / "out" / "results_rq1_trend", rng)   # a new specification
    n = 30
    (root / "data" / "interim").mkdir(parents=True)
    pd.DataFrame({
        "cps_occ": range(n), "soc_major": ["15"] * 20 + ["41"] * 10,
        "c1_nonroutine_analytic": .3, "c2_nonroutine_interpersonal": .3,
        "c3_routine_cognitive": .2, "c4_residual": .2, "ln_T": 4.8,
        "exposure": rng.uniform(0, 1, n), "emp_total": rng.uniform(1e4, 1e5, n),
        "z1": 0.1, "z2": 0.0, "z3": 0.3,
        "exposure_tercile": ["low", "mid", "high"] * 10}).to_csv(
        root / "data" / "interim" / "occ_measures.csv", index=False)
    (root / "logs").mkdir()
    (root / "logs" / "05_sample.log").write_text(SAMPLE_LOG)
    (root / "logs" / "provenance.txt").write_text("commit: abc1234\n")
    return root


def test_constants_match_pipeline():
    est = load("07_estimate")
    sample = load("05_build_sample")
    assert fig.REF_Q == est.REF_Q
    assert fig.KNOWLEDGE_MAJOR == sample.KNOWLEDGE_MAJOR


def test_sample_log_parsed_into_table_4_1(tmp_path):
    p = tmp_path / "05_sample.log"
    p.write_text(SAMPLE_LOG)
    t = fig.parse_sample_log(p)
    assert list(t["records"]) == [1000, 400, 390, 90, 300]
    assert list(t["dropped"][:3].astype("float").fillna(-1)) == [-1, 600, 10]


def test_specs_discovered_main_first_and_new_ones_included(run_root):
    specs = fig.discover_specs(run_root)
    assert specs == ["results", "results_drop_pandemic", "results_rq1_trend"]
    assert fig.label_for("results_rq1_trend", fig.SPEC_LABELS) == "Rq1 trend"
    assert fig.discover_specs(run_root, ["results_rq1_trend"]) == ["results_rq1_trend"]
    with pytest.raises(SystemExit):
        fig.discover_specs(run_root, ["results_missing"])


def test_reference_quarter_inserted_as_zero():
    es = _event("expJq", ["2022Q2", "2022Q3", "2022Q4"], np.random.default_rng(0))
    out = fig.with_reference(es)
    assert list(out["quarter"]) == ["2022Q2", "2022Q3", "2022Q4"]
    assert out.loc[out["quarter"] == fig.REF_Q, "estimate"].item() == 0.0


def test_stars_and_cells():
    assert fig.stars(0.005) == "***" and fig.stars(0.03) == "**"
    assert fig.stars(0.07) == "*" and fig.stars(0.2) == "" and fig.stars(np.nan) == ""
    assert fig.coef_cells(0.12345, 0.0456, 0.02) == ("0.123**", "(0.046)")


def test_full_build_writes_every_output(run_root, tmp_path):
    out = tmp_path / "chapter4"
    specs = fig.discover_specs(run_root)
    labels = dict(fig.SPEC_LABELS, results_rq1_trend="RQ1 with trends")
    assert fig.build(run_root, out, specs, labels) == 0
    figs = {p.name for p in (out / "figures").glob("*.png")}
    for name in ["fig_rq1_event_study", "fig_rq1_event_study_by_spec", "fig_rq2_event_study",
                 "fig_rq4_earnings", "fig_rq3_task_prices", "fig_exposure_distribution",
                 "fig_task_composition_by_tercile"]:
        assert f"{name}.png" in figs and (out / "figures" / f"{name}.pdf").exists()
    tables = {p.stem for p in (out / "tables").glob("*.csv")}
    assert {"table_4_1_sample", "table_4_2_descriptives", "table_rq1_event_study",
            "table_rq3_hedonic", "table_rq4_decomposition", "table_pretrends"} <= tables
    rq1 = (out / "tables" / "table_rq1_event_study.md").read_text()
    assert "RQ1 with trends" in rq1 and "Pandemic window dropped" in rq1
    # descriptives use knowledge-intensive occupations only (20 of 30)
    desc = pd.read_csv(out / "tables" / "table_4_2_descriptives.csv")
    assert desc["Occupations"].sum() == 20
    assert "commit: abc1234" in (out / "SOURCES.txt").read_text()
    xl = pd.ExcelFile(out / "chapter4_tables.xlsx")
    assert "4_1_sample" in xl.sheet_names and "rq3_hedonic" in xl.sheet_names
    tex = (out / "tables" / "table_rq3_hedonic.tex").read_text()
    assert r"\toprule" in tex and "_" not in tex.replace(r"\_", "")


def test_cli_runs_without_logs_or_occupation_file(run_root, tmp_path):
    (run_root / "logs" / "05_sample.log").unlink()
    (run_root / "data" / "interim" / "occ_measures.csv").unlink()
    out = tmp_path / "c4"
    assert fig.build(run_root, out, ["results"], fig.SPEC_LABELS) == 0
    assert (out / "figures" / "fig_rq1_event_study.png").exists()
    assert not (out / "tables" / "table_4_1_sample.csv").exists()
