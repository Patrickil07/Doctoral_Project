"""
Unit and smoke tests for the pipeline scripts in src/.

These use tiny hand-made inputs to check the maths and the code paths. They
never touch real data and must never be used to produce results.

Run:  pytest -q
"""
import importlib.util
import pathlib
import subprocess
import sys
import urllib.parse

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAP = ROOT / "mapping" / "onet_activity_map.csv"


def load(stem: str):
    """Import a src/NN_name.py script as a module (names start with digits)."""
    spec = importlib.util.spec_from_file_location(stem, SRC / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conv = load("00b_convert_exposure")
task = load("02_build_task_composition")
xwalk = load("03_crosswalk")
ipums = load("04_ipums_extract")
sample = load("05_build_sample")
est = load("07_estimate")


# --- mapping -----------------------------------------------------------------
def test_mapping_is_complete_and_unique():
    mp = pd.read_csv(MAP)
    assert list(mp.columns) == ["element_name", "task_part"]
    assert not mp["element_name"].duplicated().any()
    assert set(mp["task_part"]) <= set(task.PARTS)
    assert set(mp["task_part"]) == set(task.PARTS), "every part needs at least one GWA"


# --- compositional maths -----------------------------------------------------
def test_zero_replacement_closes_and_preserves_ratios():
    X = np.array([[2.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    C = task.multiplicative_replacement(X, delta=1e-3)
    np.testing.assert_allclose(C.sum(axis=1), 1.0)
    assert C[0, 1] == pytest.approx(1e-3)
    assert C[0, 0] / C[0, 2] == pytest.approx(2.0)          # non-zero ratios kept
    np.testing.assert_allclose(C[1], 0.25)


def test_ilr_zero_at_barycentre_and_is_isometric():
    Z = task.ilr_balances(np.full((1, 4), 0.25))
    np.testing.assert_allclose(Z, 0.0, atol=1e-12)
    # ILR is an isometry: Euclidean distance in z equals Aitchison distance
    rng = np.random.default_rng(0)
    C = rng.dirichlet(np.ones(4), size=2)
    clr = np.log(C) - np.log(C).mean(axis=1, keepdims=True)
    Z = task.ilr_balances(C)
    assert np.linalg.norm(Z[0] - Z[1]) == pytest.approx(np.linalg.norm(clr[0] - clr[1]))


def test_crosswalk_ilr_matches_step02():
    """Step 03 recomputes ILR inline; it must agree with step 02's function."""
    rng = np.random.default_rng(1)
    C = rng.dirichlet(np.ones(4), size=5)
    c1, c2, c3, c4 = C.T
    z = np.column_stack([np.sqrt(3 / 4) * np.log(np.cbrt(c1 * c2 * c3) / c4),
                         np.sqrt(2 / 3) * np.log(np.sqrt(c1 * c3) / c2),
                         np.sqrt(1 / 2) * np.log(c1 / c3)])
    np.testing.assert_allclose(z, task.ilr_balances(C))


# --- step 02 end to end on a fake O*NET file ----------------------------------
def _fake_onet(tmp: pathlib.Path, names: list[str]) -> pathlib.Path:
    rows = []
    for soc, base in (("15-1252.00", 3.0), ("43-9061.00", 2.0)):
        for i, n in enumerate(names):
            rows.append({"O*NET-SOC Code": soc, "Element ID": f"4.A.{i}",
                         "Element Name": n, "Scale ID": "IM",
                         "Data Value": base + (i % 3) * 0.5})
    d = tmp / "onet"
    d.mkdir()
    pd.DataFrame(rows).to_csv(d / "Work Activities.txt", sep="\t", index=False)
    return d


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SRC / script), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_step02_runs_with_committed_mapping(tmp_path):
    names = pd.read_csv(MAP)["element_name"].tolist()
    onet = _fake_onet(tmp_path, names)
    out = tmp_path / "tc.csv"
    before = MAP.read_bytes()
    r = _run("02_build_task_composition.py", "--onet", str(onet), "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert MAP.read_bytes() == before, "step 02 must never rewrite the mapping"
    df = pd.read_csv(out)
    assert len(df) == 2
    np.testing.assert_allclose(df[task.PARTS].sum(axis=1), 1.0)
    assert set(df["soc2018"]) == {"15-1252", "43-9061"}


def test_step02_fails_loudly_on_unknown_mapping_entry(tmp_path):
    names = pd.read_csv(MAP)["element_name"].tolist()
    onet = _fake_onet(tmp_path, names[1:])            # O*NET lacks one mapped element
    r = _run("02_build_task_composition.py", "--onet", str(onet),
             "--out", str(tmp_path / "tc.csv"))
    assert r.returncode == 2
    assert "MAPPING ERROR" in r.stderr


# --- step 03 -------------------------------------------------------------------
def test_wildcard_expansion():
    xw = pd.DataFrame({"cps_occ": [800, 1005], "soc_pattern": ["13-20XX", "15-1252"]})
    out = xwalk.expand_wildcards(xw, pd.Series(["13-2011", "13-2051", "13-1111", "15-1252"]))
    got = set(map(tuple, out.to_numpy().tolist()))
    assert got == {(800, "13-2011"), (800, "13-2051"), (1005, "15-1252")}


def test_broad_soc_codes_expand_like_wildcards():
    xw = pd.DataFrame({"cps_occ": [1050, 1010], "soc_pattern": ["15-1230", "15-1251"]})
    out = xwalk.expand_wildcards(xw, pd.Series(["15-1231", "15-1232", "15-1251"]))
    got = set(map(tuple, out.to_numpy().tolist()))
    assert got == {(1050, "15-1231"), (1050, "15-1232"), (1010, "15-1251")}


def test_crosswalk_reader_skips_section_header_rows(tmp_path):
    """Census list rows like '0010-0440 | 11-0000' are headings, not occupations."""
    rows = [["2018 Census Occupation Code List", None, None],
            ["2018 Census Title", "2018 Census Code", "2018 SOC Code"],
            ["Management Occupations:", "0010-0440", "11-0000"],
            ["Chief executives", "0010", "11-1011"],
            ["General and operations managers", "0020", "11-1021"],
            ["Legislators", "0030", "11-1031"],
            ["Computer and mathematical occupations:", "1005-1240", "15-0000"],
            ["Software developers", "1021", "15-1252"],
            ["Database administrators and architects", "1065", "15-124X"]]
    f = tmp_path / "xw.xlsx"
    pd.DataFrame(rows).to_excel(f, header=False, index=False)
    xw = xwalk.load_crosswalk(f)
    assert set(xw["cps_occ"]) == {10, 20, 30, 1021, 1065}
    assert not xw["soc_pattern"].str.endswith("0000").any()


# --- step 00b -----------------------------------------------------------------
def test_exposure_conversion_split_and_merge(tmp_path):
    # BLS layout: title rows above the header
    rows = [["2010 to 2018 SOC Crosswalk", None, None, None],
            [None, None, None, None],
            ["2010 SOC Code", "2010 SOC Title", "2018 SOC Code", "2018 SOC Title"],
            ["15-1132", "Software Developers, Applications", "15-1252", "Software Developers"],
            ["15-1133", "Software Developers, Systems", "15-1252", "Software Developers"],
            ["15-1143", "Computer Network Architects", "15-1241", "Network Architects"],
            ["15-1141", "Database Administrators", "15-1242", "Database Administrators"],
            ["15-1141", "Database Administrators", "15-1243", "Database Architects"],
            ["11-1011", "Chief Executives", "11-1011", "Chief Executives"]]
    f = tmp_path / "soc.xlsx"
    pd.DataFrame(rows).to_excel(f, header=False, index=False)
    xw = conv.load_soc_crosswalk(f)
    exp = pd.DataFrame({"soc": ["15-1132", "15-1133", "15-1141", "11-1011", "15-1143"],
                        "exposure": [0.8, 1.2, 1.0, 1.3, 0.5]})
    out = conv.convert(exp, xw).set_index("soc2018")
    assert out.loc["15-1252", "exposure"] == pytest.approx(1.0)     # merge -> mean
    assert out.loc["15-1252", "n_source"] == 2
    assert out.loc["15-1242", "exposure"] == out.loc["15-1243", "exposure"] == 1.0  # split
    assert out.loc["11-1011", "exposure"] == 1.3


def test_eloundou_onet_soc_codes_collapse_to_six_digit(tmp_path):
    f = tmp_path / "occ_level.csv"
    pd.DataFrame({
        "O*NET-SOC Code": ["15-1211.00", "15-1211.01", "15-1252.00", "13-2011.00"],
        "Title": ["Systems Analysts", "Health Informatics", "Software Dev", "Accountants"],
        "dv_rating_beta": [0.75, 0.48, 0.87, 0.56],
        "human_rating_beta": [0.40, 0.60, 0.45, 0.52]}).to_csv(f, index=False)
    out = conv.load_exposure(f, "human_rating_beta").set_index("soc")
    assert list(out.index) == ["13-2011", "15-1211", "15-1252"]
    assert out.loc["15-1211", "exposure"] == pytest.approx(0.50)     # mean of detail
    assert out.loc["15-1211", "n_source"] == 2
    gpt4 = conv.load_exposure(f, "dv_rating_beta").set_index("soc")
    assert gpt4.loc["15-1252", "exposure"] == pytest.approx(0.87)
    with pytest.raises(ValueError):
        conv.load_exposure(f, "no_such_column")


def test_step00b_default_run_writes_soc2018_file(tmp_path):
    src = tmp_path / "occ_level.csv"
    pd.DataFrame({"O*NET-SOC Code": ["15-1252.00", "15-1253.00"],
                  "human_rating_beta": [0.45, 0.61]}).to_csv(src, index=False)
    out = tmp_path / "exposure_soc2018.csv"
    r = _run("00b_convert_exposure.py", "--exposure", str(src), "--out", str(out))
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out, dtype={"soc2018": str})
    assert list(df.columns) == ["soc2018", "exposure", "n_source"]
    assert set(df["soc2018"]) == {"15-1252", "15-1253"}


def test_cpi_parse_keeps_unpublished_month_blank():
    fetch = load("00_fetch_public_inputs")
    rows = ["series_id        \tyear\tperiod\tvalue\tfootnote_codes",
            "CUUR0000SA0      \t2025\tM09\t 324.800\t",
            "CUUR0000SA0      \t2025\tM10\t -\t",
            "CUUR0000SA0      \t2025\tM11\t 325.000\t",
            "CUUR0000SA0      \t2025\tM13\t 323.000\t",
            "CUUR0000AA0      \t2025\tM09\t 999.000\t"]
    out = fetch.parse_cpi("\n".join(rows).encode())
    assert list(out["month"]) == [9, 10, 11]
    assert out["cpi"].isna().tolist() == [False, True, False]
    assert out["cpi"].iloc[0] == pytest.approx(324.8)


# --- step 04 -------------------------------------------------------------------
def test_month_samples():
    s = ipums.month_samples("2019-01", "2025-12")
    assert len(s) == 84 and s[0] == "cps2019_01b" and s[-1] == "cps2025_12b"


# --- step 05 -------------------------------------------------------------------
def test_sample_starts_january_2020():
    d = pd.DataFrame({"YEAR": [2019, 2019, 2020, 2020, 2025],
                      "MONTH": [1, 12, 1, 6, 12]})
    kept = sample.restrict_period(d)
    assert list(zip(kept.YEAR, kept.MONTH)) == [(2020, 1), (2020, 6), (2025, 12)]
    assert len(sample.restrict_period(d, "2020-06")) == 2
    with pytest.raises(ValueError):
        sample.restrict_period(d, "2019-01")


# --- step 07 smoke test: the specifications estimate without error ----------
@pytest.fixture(scope="module")
def fake_sample() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    quarters = [str(p) for p in pd.period_range("2021Q1", "2023Q4", freq="Q")]
    occs = np.arange(10, 30)
    exposure = dict(zip(occs, rng.uniform(0, 1, len(occs))))
    n = 6000
    d = pd.DataFrame({
        "OCC": rng.choice(occs, n), "quarter": rng.choice(quarters, n),
        "STATEFIP": rng.choice([6, 36, 48], n), "IND": rng.choice([7380, 7390], n),
        "SEX": rng.choice([1, 2], n), "AGE": rng.integers(22, 56, n),
        "EDUC": rng.choice([73, 111, 123], n), "EARNWT": rng.uniform(500, 3000, n)})
    d["exposure"] = d["OCC"].map(exposure)
    d["early_career"] = (d["AGE"] <= 30).astype(int)
    d["post"] = (d["quarter"] >= "2022Q4").astype(int)
    for z in ("z1", "z2", "z3", "ln_T"):
        d[z] = d["OCC"].map(dict(zip(occs, rng.normal(size=len(occs))))) + rng.normal(0, .1, n)
    d["ln_w"] = 7 + 0.1 * d["z3"] + rng.normal(0, 0.3, n)
    return d


def test_estimation_specs_run(fake_sample):
    rq1 = est.event_study(fake_sample, "z3", with_tasks=False)
    assert est.REF_Q not in set(rq1["quarter"]) and len(rq1) == 11
    assert est.pretrend_test(rq1)["n_pre"] > 0
    rq3 = est.hedonic(fake_sample)
    assert rq3["term"].str.contains("z3:early_career:post").any()
    rq2 = est.early_share(fake_sample)
    assert len(rq2) == 11 and rq2["se"].notna().all()
