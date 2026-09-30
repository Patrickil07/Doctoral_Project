"""
Step 08 (src/08_honest_did.R) on made-up event-study coefficients.

Skipped when R or the HonestDiD package is not installed (the tests workflow
installs both). Nothing here is a result.
"""
import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _have_honestdid() -> bool:
    if not shutil.which("Rscript"):
        return False
    r = subprocess.run(["Rscript", "-e", "library(HonestDiD)"], capture_output=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not _have_honestdid(), reason="R package HonestDiD not installed")


def _write(folder: pathlib.Path, pre: list, post: list, se: float):
    qs = pd.period_range("2020Q1", "2025Q4", freq="Q").astype(str)
    pre_q = [q for q in qs if q < "2022Q3"][-len(pre):]
    post_q = [q for q in qs if q > "2022Q3"][:len(post)]
    es = pd.DataFrame({"term": [f"expJq_{q}" for q in pre_q + post_q],
                       "estimate": pre + post, "se": se, "pvalue": 0.5,
                       "quarter": pre_q + post_q,
                       "period": ["pre"] * len(pre) + ["post"] * len(post)})
    folder.mkdir(parents=True, exist_ok=True)
    es.to_csv(folder / "rq1_task_composition.csv", index=False)
    V = pd.DataFrame(np.eye(len(es)) * se ** 2, index=es["term"], columns=es["term"])
    V.to_csv(folder / "rq1_task_composition_vcov.csv")


def _run(folder: pathlib.Path) -> pd.DataFrame:
    r = subprocess.run(["Rscript", str(ROOT / "src" / "08_honest_did.R"), "--results",
                        str(folder), "--mbar-max", "1", "--mbar-step", "0.5"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return pd.read_csv(folder / "rq1_task_composition_honest_did.csv")


def test_clear_break_with_flat_pretrend_survives_some_mbar(tmp_path):
    _write(tmp_path, [0.001, -0.001, 0.002, 0.0], [0.5, 0.5, 0.5, 0.5], se=0.01)
    out = _run(tmp_path)
    orig = out[out["method"] == "Original"].iloc[0]
    assert orig["lb"] > 0 and bool(orig["excludes_zero"])
    assert out["breakdown_Mbar"].iloc[0] >= 0.5
    rob = out[out["method"] != "Original"].sort_values("Mbar")
    widths = (rob["ub"] - rob["lb"]).to_numpy()
    assert np.isfinite(widths).all() and widths[-1] > widths[0]   # widen with Mbar
    assert rob["lb"].iloc[0] < 0.5 < rob["ub"].iloc[0]


def test_no_effect_has_no_breakdown(tmp_path):
    _write(tmp_path, [0.01, -0.01, 0.0], [0.0, 0.01, -0.01], se=0.02)
    out = _run(tmp_path)
    assert pd.isna(out["breakdown_Mbar"].iloc[0])


def test_gapped_quarters_are_refused(tmp_path):
    _write(tmp_path, [0.0, 0.0, 0.0], [0.1, 0.1], se=0.01)
    es = pd.read_csv(tmp_path / "rq1_task_composition.csv").drop(index=1)
    es.to_csv(tmp_path / "rq1_task_composition.csv", index=False)
    r = subprocess.run(["Rscript", str(ROOT / "src" / "08_honest_did.R"), "--results",
                        str(tmp_path)], capture_output=True, text=True)
    assert r.returncode != 0 and "consecutive" in r.stderr
