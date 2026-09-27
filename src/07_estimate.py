"""
07_estimate.py — estimate RQ1-RQ4 exactly as specified in proposal Section 6.5.

Equations:
  (5) RQ1  z3(k) = sum_q beta_q (E_k x 1[t=q]) + sum_q gamma_q (E_k x J_i x 1[t=q])
                   + X'd + tau_st + rho (J_i x tau_t) + e
  (6) RQ2  EarlyShare_kst = sum_q beta_q (E_k x 1[t=q]) + eta_k + tau_st + e
  (7) RQ3  ln W = sum_m theta_m z_m + sum_m thetaJ_m (z_m x J) + sum_m psi_m (z_m x J x Post)
                   + lambda ln T + X'd + tau_st + rho (J x Post) + e
  RQ4      Eq (5) re-run with ln W as outcome, with and without task controls;
           the difference decomposes composition vs price.

Fixed effects: tau_st is state-by-quarter in all equations; RQ1 and RQ4 also
include early-career-by-quarter effects (rho J_i x tau_t). In RQ1, RQ3 and RQ4
these (and industry in RQ3) are absorbed by weighted within-group demeaning
(alternating projections), which gives the same coefficients as including the
dummies without building a design matrix with >1,000 columns.

Inference: weighted by EARNWT, standard errors clustered on occupation
(exposure varies at that level). Reference quarter 2022Q3.

Usage:
    python src/07_estimate.py --sample data/out/analysis_sample.parquet \
        --out data/out/results
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

REF_Q = "2022Q3"


def _event_dummies(d: pd.DataFrame, cols: dict) -> tuple[pd.DataFrame, list]:
    """Explicit event-time interaction columns, omitting the reference quarter.

    Built by hand rather than via patsy's numeric:categorical syntax, which
    retains the reference level and so collides with occupation fixed effects
    (exposure is occupation-invariant) producing a singular design.
    """
    quarters = [q for q in sorted(d["quarter"].unique()) if q != REF_Q]
    names = []
    for q in quarters:
        ind = (d["quarter"] == q).astype(float)
        for prefix, base in cols.items():
            nm = f"{prefix}_{q}"
            d[nm] = base(d) * ind
            names.append(nm)
    return d, names


def absorb(d: pd.DataFrame, cols: list, groups: list, w: np.ndarray,
           tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    """Weighted within-transformation of `cols` for one or more fixed effects.

    Each fixed effect is swept out by subtracting its weighted group means; with
    several fixed effects the sweeps alternate until the group means are ~0
    (method of alternating projections). By Frisch-Waugh-Lovell, regressing the
    transformed outcome on the transformed regressors reproduces the
    coefficients of the regression with the dummies included.
    """
    X = d[cols].to_numpy(dtype=float).copy()
    codes = [pd.factorize(d[g])[0] for g in groups]
    wsum = [np.bincount(c, weights=w) for c in codes]
    for _ in range(max_iter):
        worst = 0.0
        for c, ws in zip(codes, wsum):
            for j in range(X.shape[1]):
                means = np.bincount(c, weights=X[:, j] * w) / ws
                worst = max(worst, float(np.abs(means).max()))
                X[:, j] -= means[c]
        if len(codes) == 1 or worst < tol:
            break
    return X


def _fit_absorbed(d: pd.DataFrame, outcome: str, regressors: list, groups: list,
                  cluster: str):
    w = d["EARNWT"].to_numpy(dtype=float)
    Z = absorb(d, [outcome] + regressors, groups, w)
    m = sm.WLS(Z[:, 0], Z[:, 1:], weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    return (pd.Series(m.params, index=regressors), pd.Series(m.bse, index=regressors),
            pd.Series(m.pvalues, index=regressors))


def _state_quarter(d: pd.DataFrame) -> pd.Series:
    return d["STATEFIP"].astype(str) + "_" + d["quarter"].astype(str)


def event_study(df: pd.DataFrame, outcome: str, with_tasks: bool,
                cluster: str = "OCC") -> pd.DataFrame:
    d = df.copy()
    d, names = _event_dummies(d, {
        "expq": lambda x: x["exposure"],
        "expJq": lambda x: x["exposure"] * x["early_career"],
    })
    d["state_quarter"] = _state_quarter(d)
    # rho (J_i x tau_t): early-career x quarter effects absorb economy-wide shifts
    # in the early-career gap; the early_career main effect is nested in them.
    d["early_quarter"] = d["early_career"].astype(int).astype(str) + "_" + d["quarter"].astype(str)
    regs = names + ["SEX", "AGE", "EDUC"]
    if with_tasks:
        regs += ["z1", "z2", "z3", "ln_T"]
    b, se, p = _fit_absorbed(d, outcome, regs, ["state_quarter", "early_quarter"], cluster)
    keep = [r for r in regs if r.startswith("expJq_")]
    out = pd.DataFrame({"term": keep, "estimate": b[keep].to_numpy(),
                        "se": se[keep].to_numpy(), "pvalue": p[keep].to_numpy()})
    out["quarter"] = out["term"].str.replace("expJq_", "", regex=False)
    out["period"] = np.where(out["quarter"] < REF_Q, "pre", "post")
    return out.sort_values("quarter").reset_index(drop=True)


def pretrend_test(res: pd.DataFrame) -> dict:
    """Joint significance of pre-period interaction coefficients."""
    pre = res[res["period"] == "pre"] if "period" in res.columns else res.iloc[0:0]
    if pre.empty:
        return {"n_pre": 0, "warning": "no pre-period coefficients parsed"}
    z = pre["estimate"] / pre["se"]
    return {"n_pre": len(pre), "max_abs_z": float(z.abs().max()),
            "any_sig_5pct": bool((pre["pvalue"] < 0.05).any())}


def hedonic(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["state_quarter"] = _state_quarter(d)
    regs = ["z1", "z2", "z3", "ln_T"]
    for z in ("z1", "z2", "z3"):
        d[f"{z}:early_career"] = d[z] * d["early_career"]
        d[f"{z}:early_career:post"] = d[z] * d["early_career"] * d["post"]
    regs += [f"{z}:early_career" for z in ("z1", "z2", "z3")]
    regs += [f"{z}:early_career:post" for z in ("z1", "z2", "z3")]
    d["early_career:post"] = d["early_career"] * d["post"]
    regs += ["early_career:post", "early_career", "SEX", "AGE", "EDUC"]
    b, se, p = _fit_absorbed(d, "ln_w", regs, ["state_quarter", "IND"], "OCC")
    keep = [r for r in regs if r.startswith(("z1", "z2", "z3", "ln_T"))
            or "early_career:post" in r]
    return pd.DataFrame({"term": keep, "estimate": b[keep].to_numpy(),
                         "se": se[keep].to_numpy(), "pvalue": p[keep].to_numpy()})


def early_share(df: pd.DataFrame) -> pd.DataFrame:
    cell = (df.groupby(["OCC", "STATEFIP", "quarter"])
              .apply(lambda g: pd.Series({
                  "early_share": np.average(g["early_career"], weights=g["EARNWT"]),
                  "exposure": g["exposure"].iloc[0],
                  "n": len(g), "w": g["EARNWT"].sum()}), include_groups=False)
              .reset_index())
    cell, names = _event_dummies(cell, {"expq": lambda x: x["exposure"]})
    base = "early_share ~ " + " + ".join(names) + " + C(OCC) + C(quarter)"
    try:
        m = smf.wls(base + " + C(STATEFIP):C(quarter)", data=cell,
                    weights=cell["w"]).fit(cov_type="cluster",
                                           cov_kwds={"groups": cell["OCC"]})
        if np.linalg.matrix_rank(m.model.exog) < m.model.exog.shape[1]:
            raise np.linalg.LinAlgError("rank deficient")
        spec = "state-by-quarter FE"
    except (np.linalg.LinAlgError, ValueError):
        m = smf.wls(base + " + C(STATEFIP)", data=cell, weights=cell["w"]).fit(
            cov_type="cluster", cov_kwds={"groups": cell["OCC"]})
        spec = "additive state + quarter FE (state-by-quarter rank-deficient)"
    print(f"      RQ2 specification: {spec}")
    keep = [p for p in m.params.index if p.startswith("expq_")]
    out = pd.DataFrame({"term": keep, "estimate": m.params[keep].to_numpy(),
                        "se": m.bse[keep].to_numpy(),
                        "pvalue": m.pvalues[keep].to_numpy()})
    out["quarter"] = out["term"].str.replace("expq_", "", regex=False)
    out["period"] = np.where(out["quarter"] < REF_Q, "pre", "post")
    return out.sort_values("quarter").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/out/analysis_sample.parquet")
    ap.add_argument("--out", default="data/out/results")
    ap.add_argument("--drop-pandemic", action="store_true")
    args = ap.parse_args()

    sample = pathlib.Path(args.sample)
    if not sample.exists():
        print(f"[est] {sample} not found.\n"
              "      Estimation needs the CPS analysis sample, which needs:\n"
              "        step 04  IPUMS extract   (requires CPS registration + API key)\n"
              "        step 05  sample build    (requires step 03 occupation measures)\n"
              "      Nothing to estimate yet - this is expected, not a bug.")
        return 1
    df = pd.read_parquet(sample)
    if args.drop_pandemic:
        df = df[df["pandemic_window"] == 0]
        print(f"[est] pandemic window dropped, N={len(df):,}")

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[est] RQ1 — task composition of employment held (Eq. 5)")
    rq1 = event_study(df, "z3", with_tasks=False)
    rq1.to_csv(outdir / "rq1_task_composition.csv", index=False)
    print("      pre-trend check:", pretrend_test(rq1))

    print("[est] RQ2 — early-career employment share (Eq. 6)")
    rq2 = early_share(df)
    rq2.to_csv(outdir / "rq2_early_share.csv", index=False)

    print("[est] RQ3 — hedonic implicit task prices (Eq. 7)")
    rq3 = hedonic(df)
    rq3.to_csv(outdir / "rq3_hedonic.csv", index=False)
    psi3 = rq3[rq3["term"].str.contains("z3") & rq3["term"].str.contains("post")]
    if not psi3.empty:
        print(f"      psi_3 (H3 focal) = {psi3['estimate'].iloc[0]:+.4f} "
              f"(se {psi3['se'].iloc[0]:.4f}, p={psi3['pvalue'].iloc[0]:.4f})")

    print("[est] RQ4 — earnings levels, unconditional vs conditional")
    uncond = event_study(df, "ln_w", with_tasks=False)
    cond = event_study(df, "ln_w", with_tasks=True)
    uncond.to_csv(outdir / "rq4_earnings_unconditional.csv", index=False)
    cond.to_csv(outdir / "rq4_earnings_conditional.csv", index=False)
    merged = uncond.merge(cond, on="term", suffixes=("_uncond", "_cond"))
    merged["composition_component"] = (merged["estimate_uncond"]
                                       - merged["estimate_cond"])
    merged.to_csv(outdir / "rq4_decomposition.csv", index=False)
    print("      composition vs price decomposition written")

    print(f"\n[est] all results -> {outdir}")
    print("[est] paste these into Chapter 4 tables, replacing [pending] cells.")
    print("[est] run again with --drop-pandemic for the Section 3.6 robustness row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
