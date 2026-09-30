"""
07_estimate.py — estimate RQ1-RQ4 (Eqs. 5-7 of the proposal methods chapter).

Equations:
  (5) RQ1  z3(k) = a E_k + b (E_k x J_i) + sum_q beta_q (E_k x 1[t=q])
                   + sum_q gamma_q (E_k x J_i x 1[t=q]) + X'd + tau_st + rho (J_i x tau_t) + e
  (6) RQ2  EarlyShare_kst = sum_q beta_q (E_k x 1[t=q]) + eta_k + tau_st + e
  (7) RQ3  ln W = sum_m theta_m z_m + sum_m thetaJ_m (z_m x J) + sum_m pi_m (z_m x Post)
                   + sum_m psi_m (z_m x J x Post) + lambda ln T + lambdaP (ln T x Post)
                   + X'd + tau_st + rho (J x Post) + e
  RQ4      Eq (5) re-run with ln W as outcome, with and without task controls;
           the difference decomposes composition vs price.

Fixed effects: tau_st is state-by-quarter in all equations; RQ1 and RQ4 also
include early-career-by-quarter effects (rho J_i x tau_t). In RQ1, RQ3 and RQ4
these (and industry in RQ3) are absorbed by weighted within-group demeaning
(alternating projections), which gives the same coefficients as including the
dummies without building a design matrix with >1,000 columns.

Controls: SEX and AGE enter linearly; education enters as fixed effects (IPUMS
EDUC codes are categories, not years), absorbed like the other fixed effects.

Inference: RQ1, RQ3 and RQ4 weighted by EARNWT on the ORG earnings sample;
RQ2 weighted by WTFINL on all employed records of every rotation group (step
05's employment sample). Standard errors clustered on occupation (exposure
varies at that level); p-values use the normal distribution and the joint
tests chi-squared, adequate with about 180 clusters. Reference quarter 2022Q3;
post = 2022Q4 onwards in every equation.

RQ1 specification menu (see README):
  Model 1  baseline event study (default run).
  Model 2  --rq1-trend: adds a linear exposure x early-career trend
           (E_k x J_i x t, t = quarters from 2022Q3) and keeps event dummies
           for post quarters only, so the trend is fitted on the pre-period and
           each post coefficient is the deviation from its extrapolation
           (parametric event study as in Dobkin et al. 2018).
  Model 3  --start-quarter 2021Q4: estimation window starts after the pandemic.
Each event study also saves its clustered covariance matrix (<file>_vcov.csv),
used for the joint pre-trend test and by step 08 (Rambachan & Roth 2023).

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
            pd.Series(m.pvalues, index=regressors),
            pd.DataFrame(m.cov_params(), index=regressors, columns=regressors))


def _state_quarter(d: pd.DataFrame) -> pd.Series:
    return d["STATEFIP"].astype(str) + "_" + d["quarter"].astype(str)


def quarters_from_ref(q: pd.Series) -> pd.Series:
    """Event time t in quarters (2022Q3 = 0, 2022Q4 = 1, 2022Q2 = -1)."""
    ref = pd.Period(REF_Q, freq="Q")
    return q.map(lambda x: (pd.Period(x, freq="Q") - ref).n).astype(float)


def event_study(df: pd.DataFrame, outcome: str, with_tasks: bool,
                cluster: str = "OCC", trend: bool = False,
                return_vcov: bool = False):
    d = df.copy()
    d, names = _event_dummies(d, {
        "expq": lambda x: x["exposure"],
        "expJq": lambda x: x["exposure"] * x["early_career"],
    })
    if trend:
        # Model 2: E x J x t replaces the pre-period E x J dummies (with all of
        # them kept the trend would be collinear with the dummies).
        names = [n for n in names
                 if not (n.startswith("expJq_") and n.removeprefix("expJq_") < REF_Q)]
        d["expJ_trend"] = d["exposure"] * d["early_career"] * quarters_from_ref(d["quarter"])
        names.append("expJ_trend")
    # Main effects of the interacted variables. Without them, dropping the
    # 2022Q3 dummies does not normalise anything: it forces the 2022Q3 gradient
    # to zero, and every other coefficient picks up the time-invariant
    # cross-occupation gradient (E, and E x J) as if it were an event effect.
    # Occupation FE would absorb E but are infeasible for RQ1 (z3 is fixed per
    # occupation), so the two terms enter explicitly.
    d["exposure_main"] = d["exposure"]
    d["exposure_x_early"] = d["exposure"] * d["early_career"]
    names = names + ["exposure_main", "exposure_x_early"]
    d["state_quarter"] = _state_quarter(d)
    # rho (J_i x tau_t): early-career x quarter effects absorb economy-wide shifts
    # in the early-career gap; the early_career main effect is nested in them.
    d["early_quarter"] = d["early_career"].astype(int).astype(str) + "_" + d["quarter"].astype(str)
    regs = names + ["SEX", "AGE"]
    if with_tasks:
        regs += ["z1", "z2", "z3", "ln_T"]
    b, se, p, V = _fit_absorbed(d, outcome, regs, ["state_quarter", "early_quarter", "EDUC"],
                                cluster)
    keep = [r for r in regs if r.startswith("expJq_")]
    out = pd.DataFrame({"term": keep, "estimate": b[keep].to_numpy(),
                        "se": se[keep].to_numpy(), "pvalue": p[keep].to_numpy()})
    out["quarter"] = out["term"].str.replace("expJq_", "", regex=False)
    out["period"] = np.where(out["quarter"] < REF_Q, "pre", "post")
    out = out.sort_values("quarter").reset_index(drop=True)
    if trend:
        out = pd.concat([out, pd.DataFrame({
            "term": ["expJ_trend"], "estimate": [b["expJ_trend"]], "se": [se["expJ_trend"]],
            "pvalue": [p["expJ_trend"]], "quarter": [""], "period": ["trend"]})],
            ignore_index=True)
    if return_vcov:
        terms = list(out["term"])
        return out, V.loc[terms, terms]
    return out


def joint_wald(res: pd.DataFrame, vcov: pd.DataFrame, period: str = "pre") -> dict:
    """Wald test that all `period` coefficients are jointly zero (chi-squared,
    clustered covariance)."""
    from scipy import stats
    terms = list(res.loc[res["period"] == period, "term"])
    if not terms:
        return {}
    b = res.set_index("term").loc[terms, "estimate"].to_numpy()
    V = vcov.loc[terms, terms].to_numpy()
    w = float(b @ np.linalg.pinv(V) @ b)
    return {"wald_chi2": w, "wald_df": len(terms), "wald_p": float(stats.chi2.sf(w, len(terms)))}


def pretrend_test(res: pd.DataFrame, vcov: pd.DataFrame | None = None) -> dict:
    """Pre-period interaction coefficients: largest |z|, any significant, and
    (with the covariance matrix) the joint Wald test."""
    pre = res[res["period"] == "pre"] if "period" in res.columns else res.iloc[0:0]
    if pre.empty:
        return {"n_pre": 0, "warning": "no pre-period coefficients parsed"}
    z = pre["estimate"] / pre["se"]
    out = {"n_pre": len(pre), "max_abs_z": float(z.abs().max()),
           "any_sig_5pct": bool((pre["pvalue"] < 0.05).any())}
    if vcov is not None:
        out.update(joint_wald(res, vcov))
    return out


def hedonic(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["state_quarter"] = _state_quarter(d)
    regs = ["z1", "z2", "z3", "ln_T"]
    for z in ("z1", "z2", "z3"):
        d[f"{z}:early_career"] = d[z] * d["early_career"]
        d[f"{z}:early_career:post"] = d[z] * d["early_career"] * d["post"]
    regs += [f"{z}:early_career" for z in ("z1", "z2", "z3")]
    regs += [f"{z}:early_career:post" for z in ("z1", "z2", "z3")]
    # Task x Post (and ln T x Post): lower-order terms of the psi interactions.
    # Without them psi_m also absorbs any post-2022 change in the price of z_m
    # that is common to early-career and experienced workers.
    for z in ("z1", "z2", "z3", "ln_T"):
        d[f"{z}:post"] = d[z] * d["post"]
    regs += [f"{z}:post" for z in ("z1", "z2", "z3", "ln_T")]
    d["early_career:post"] = d["early_career"] * d["post"]
    regs += ["early_career:post", "early_career", "SEX", "AGE"]
    b, se, p, _ = _fit_absorbed(d, "ln_w", regs, ["state_quarter", "IND", "EDUC"], "OCC")
    keep = [r for r in regs if r.startswith(("z1", "z2", "z3", "ln_T"))
            or "early_career:post" in r]
    return pd.DataFrame({"term": keep, "estimate": b[keep].to_numpy(),
                         "se": se[keep].to_numpy(), "pvalue": p[keep].to_numpy()})


def early_share(df: pd.DataFrame, return_vcov: bool = False, weight: str = "EARNWT"):
    """Eq. 6 on occupation x state x quarter cells. `df` is step 05's employment
    sample (all rotation groups, weight WTFINL) when available."""
    g = df.assign(_w=df[weight].astype(float), _we=df[weight].astype(float) * df["early_career"])
    cell = (g.groupby(["OCC", "STATEFIP", "quarter"])
              .agg(w=("_w", "sum"), we=("_we", "sum"), exposure=("exposure", "first"),
                   n=("_w", "size"))
              .reset_index())
    cell = cell[cell["w"] > 0]
    cell["early_share"] = cell["we"] / cell["w"]
    print(f"      RQ2 cells: {len(cell):,} occupation x state x quarter cells from "
          f"{len(df):,} records (weight {weight}); median {cell['n'].median():.0f} "
          "records per cell")
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
    out = out.sort_values("quarter").reset_index(drop=True)
    if return_vcov:
        terms = list(out["term"])
        return out, m.cov_params().loc[terms, terms]
    return out


def save_event(res: pd.DataFrame, vcov: pd.DataFrame, path: pathlib.Path):
    res.to_csv(path, index=False)
    vcov.to_csv(path.with_name(path.stem + "_vcov.csv"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/out/analysis_sample.parquet")
    ap.add_argument("--employment", default=None,
                    help="RQ2 sample (default: <sample stem>_employment.parquet if present)")
    ap.add_argument("--out", default="data/out/results")
    ap.add_argument("--drop-pandemic", action="store_true")
    ap.add_argument("--start-quarter", default="",
                    help="first quarter kept, e.g. 2021Q4 (RQ1 Model 3)")
    ap.add_argument("--rq1-trend", action="store_true",
                    help="RQ1 Model 2: linear E x J x t trend, post-quarter dummies only")
    ap.add_argument("--rq1-only", action="store_true",
                    help="estimate RQ1 only (the extra RQ1 specifications)")
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
    # nullable integer columns (from older samples) break the formula parser
    nullable = [c for c in df.columns if pd.api.types.is_extension_array_dtype(df[c])
                and pd.api.types.is_numeric_dtype(df[c])]
    if nullable:
        df[nullable] = df[nullable].astype("float64")
    if args.drop_pandemic:
        df = df[df["pandemic_window"] == 0]
        print(f"[est] pandemic window dropped, N={len(df):,}")
    if args.start_quarter:
        if not args.start_quarter < REF_Q:
            raise SystemExit(f"--start-quarter must be before the reference quarter {REF_Q}")
        df = df[df["quarter"] >= args.start_quarter]
        print(f"[est] window starts {args.start_quarter}, N={len(df):,}")

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[est] RQ1 — task composition of employment held (Eq. 5)")
    rq1, v1 = event_study(df, "z3", with_tasks=False, trend=args.rq1_trend,
                          return_vcov=True)
    save_event(rq1, v1, outdir / "rq1_task_composition.csv")
    print("      pre-trend check:", pretrend_test(rq1, v1))
    if args.rq1_trend:
        t = rq1[rq1["period"] == "trend"].iloc[0]
        print(f"      linear E x J x t trend = {t['estimate']:+.4f} "
              f"(se {t['se']:.4f}, p={t['pvalue']:.4f}); post coefficients are "
              "deviations from it")
    if args.rq1_only:
        print(f"\n[est] RQ1 results -> {outdir}")
        return 0

    print("[est] RQ2 — early-career employment share (Eq. 6)")
    emp_path = pathlib.Path(args.employment or sample.with_name(sample.stem + "_employment.parquet"))
    if emp_path.exists():
        emp = pd.read_parquet(emp_path)
        if args.drop_pandemic:
            emp = emp[emp["pandemic_window"] == 0]
        if args.start_quarter:
            emp = emp[emp["quarter"] >= args.start_quarter]
        rq2, v2 = early_share(emp, return_vcov=True, weight="WTFINL")
    else:
        print(f"      WARNING: {emp_path} not found; RQ2 uses the ORG earnings sample")
        rq2, v2 = early_share(df, return_vcov=True)
    save_event(rq2, v2, outdir / "rq2_early_share.csv")

    print("[est] RQ3 — hedonic implicit task prices (Eq. 7)")
    rq3 = hedonic(df)
    rq3.to_csv(outdir / "rq3_hedonic.csv", index=False)
    psi3 = rq3[rq3["term"].str.contains("z3") & rq3["term"].str.contains("post")]
    if not psi3.empty:
        print(f"      psi_3 (H3 focal) = {psi3['estimate'].iloc[0]:+.4f} "
              f"(se {psi3['se'].iloc[0]:.4f}, p={psi3['pvalue'].iloc[0]:.4f})")

    print("[est] RQ4 — earnings levels, unconditional vs conditional")
    uncond, vu = event_study(df, "ln_w", with_tasks=False, return_vcov=True)
    cond, vc = event_study(df, "ln_w", with_tasks=True, return_vcov=True)
    save_event(uncond, vu, outdir / "rq4_earnings_unconditional.csv")
    save_event(cond, vc, outdir / "rq4_earnings_conditional.csv")
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
