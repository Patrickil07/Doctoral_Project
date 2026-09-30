"""
Parameter-recovery test: plant known effects in synthetic data, run the real
estimation script (src/07_estimate.py) and check they come back with the right
sign and roughly the right size, with flat pre-trends.

Run before trusting any real result, and again after any change to
02_build_task_composition.py or 07_estimate.py. Synthetic data only; nothing
here is a result.
"""
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
G_TASK, G_SHARE, PSI = 0.25, -0.20, 0.10
# Nuisance terms the estimators must not mistake for effects: a time-invariant
# exposure x early-career gradient (in z3 and wages) and a post-2022 change in
# the price of z3 common to all workers.
C_EJ, C_W, PI3 = -0.05, -0.10, 0.08


def make_synthetic(path: pathlib.Path, seed: int = 11, c_ej: float = C_EJ) -> None:
    rng = np.random.default_rng(seed)
    n_occ = 40
    quarters = pd.period_range("2020Q1", "2025Q4", freq="Q").astype(str).tolist()  # sample period
    occ = pd.DataFrame({
        "OCC": np.arange(1000, 1000 + n_occ),
        "exposure": rng.uniform(0, 1, n_occ),
        "z1": rng.normal(0, .5, n_occ), "z2": rng.normal(0, .5, n_occ),
        "z3b": rng.normal(0, .6, n_occ), "ln_T": rng.normal(4, .2, n_occ)})
    rows = []
    for q in quarters:
        post = 1 if q >= "2022Q4" else 0
        for _, o in occ.iterrows():
            share = np.clip(0.45 + G_SHARE * o.exposure * post, .05, .95)
            for st in range(1, 4):
                m = 40
                early = rng.binomial(1, share, m)
                z3 = (o.z3b + c_ej * o.exposure * early
                      + G_TASK * o.exposure * early * post + rng.normal(0, .05, m))
                lnw = (6.5 + .30 * z3 + PI3 * z3 * post + PSI * z3 * early * post
                       - .15 * early + C_W * o.exposure * early + rng.normal(0, .05, m))
                rows.append(pd.DataFrame({
                    "OCC": o.OCC, "exposure": o.exposure, "quarter": q, "post": post,
                    "early_career": early, "z1": o.z1, "z2": o.z2, "z3": z3,
                    "ln_T": o.ln_T, "ln_w": lnw, "EARNWT": rng.uniform(.8, 1.2, m),
                    "STATEFIP": st, "IND": rng.integers(1, 4, m),
                    "SEX": rng.integers(1, 3, m), "AGE": rng.integers(22, 56, m),
                    "EDUC": rng.integers(70, 125, m), "pandemic_window": 0}))
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> pathlib.Path:
    tmp = tmp_path_factory.mktemp("recovery")
    sample, out = tmp / "syn.parquet", tmp / "res"
    make_synthetic(sample)
    proc = subprocess.run([sys.executable, str(ROOT / "src" / "07_estimate.py"),
                           "--sample", str(sample), "--out", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


def _pre_post(path: pathlib.Path) -> tuple[float, float]:
    r = pd.read_csv(path)
    return (r[r.period == "pre"]["estimate"].mean(),
            r[r.period == "post"]["estimate"].mean())


def test_rq1_recovers_task_shift_with_flat_pretrend(results):
    # Coefficients are relative to one (noisy) reference quarter, so compare
    # the post-minus-pre change with the planted shift.
    pre, post = _pre_post(results / "rq1_task_composition.csv")
    assert abs((post - pre) - G_TASK) < 0.08


def test_rq2_recovers_early_share_drop_with_flat_pretrend(results):
    pre, post = _pre_post(results / "rq2_early_share.csv")
    assert abs(pre) < 0.08
    assert post < 0 and abs(post - G_SHARE) < 0.12


def test_rq3_recovers_psi3(results):
    r = pd.read_csv(results / "rq3_hedonic.csv")
    est = r.set_index("term")["estimate"]
    assert abs(est["z3:early_career:post"] - PSI) < 0.02   # not PSI + PI3
    assert abs(est["z3:post"] - PI3) < 0.02


def test_rq1_ignores_time_invariant_exposure_gradient(tmp_path):
    """A constant E x J gradient in z3 is not an event effect: the event-study
    coefficients must be identical with and without it (same random draws).
    Without the E and E x J main effects this fails by exactly C_EJ."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("est07", ROOT / "src" / "07_estimate.py")
    est = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(est)
    fits = []
    for c in (0.0, C_EJ):
        path = tmp_path / f"syn_{c}.parquet"
        make_synthetic(path, c_ej=c)
        fits.append(est.event_study(pd.read_parquet(path), "z3", with_tasks=False)["estimate"])
    assert np.allclose(fits[0], fits[1], atol=1e-8)
