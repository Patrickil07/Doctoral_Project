"""
05_build_sample.py — construct the estimation samples from the IPUMS extract.

Two outputs:
  analysis_sample.parquet             ORG earners (RQ1, RQ3, RQ4), weight EARNWT
  analysis_sample_employment.parquet  all employed records in every rotation
                                      group (RQ2 employment shares), weight WTFINL

Rules (proposal, Chapter 3):
  * wage/salary workers (not self-employed), employed
  * knowledge-intensive SOC major groups 13,15,17,19,23,27,43 (--majors), taken
    from the FULL Census code list (step 03 census_occ_codes.csv), so records
    in those groups whose occupation has no measures are counted and reported,
    not silently treated as non-knowledge
  * early-career (22-30) vs experienced (35-55) (--bands); 31-34 excluded from
    the primary contrast, restorable via --include-midband
  * EARNWEEK2 weekly earnings: IPUMS applies the April 2023 Census rounding to
    the earlier months too, so the series is comparable over time
  * allocated (imputed) earnings EXCLUDED — Hirsch & Schumacher (2004) match
    bias — using the IPUMS data-quality flag (restorable via --keep-allocated)
  * full-time workers (usual weekly hours at the main job, UHRSWORK1, 35 or
    more; "hours vary" excluded) so that weekly earnings compare pay, not
    hours; --hours all keeps everyone with reported hours and uses
    ln(weekly earnings / usual hours). (UHRSWORKORG is NOT used: its universe
    is hourly-paid workers only.)
  * top-codes: the nominal $2,884.61 cap was fixed until 2023-24 and replaced
    by monthly values afterwards. All real earnings are censored at ONE real
    cap, the lowest real value of $2,884.61 in the sample period, so the cap
    does not change inside the post period; is_topcoded marks the censored
    records and --drop-topcoded drops them
  * sample period January 2020 onwards: CPS switched to 2018 Census occupation
    codes in January 2020, and the occupation measures are keyed on those codes
  * CPI-U deflation to 2019 dollars
  * post = 2022Q4 onwards (same quarter-based timing as the event studies;
    ChatGPT was released on 30 November 2022, so 2022Q4 is partly pre-release)

Record counts are person-month records: an ORG respondent can appear twice
(month-in-sample 4 and 8), a basic-monthly respondent up to eight times.

Usage:
    python src/05_build_sample.py --ipums data/raw/ipums \
        --occ data/interim/occ_measures.csv --cpi data/raw/cpi_u.csv \
        --out data/out/analysis_sample.parquet
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

KNOWLEDGE_MAJOR = {"13", "15", "17", "19", "23", "27", "43"}
# Nominal weekly-earnings top-code while it was fixed (IPUMS CPS, EARNWEEK2).
TOPCODE_NOMINAL = 2884.61
POST_START = "2022-10-01"
# Usual weekly hours at the main job; 997 = hours vary, 999 = not in universe.
HOURS_VAR = "UHRSWORK1"
HOURS_MAX = 198          # 2022Q4, the first event-study post quarter
# First CPS month coded with the 2018 Census occupation classification.
SAMPLE_START = "2020-01"
# "Not in universe" codes from the IPUMS CPS codebook (Version 13.0).
EARNINGS_NIU = {"EARNWEEK": 9999.99, "EARNWEEK2": 999999.99}


def restrict_period(df: pd.DataFrame, start: str = SAMPLE_START) -> pd.DataFrame:
    """Keep records from `start` (YYYY-MM) onwards.

    Starting earlier than 2020-01 is refused: those records carry 2010 Census
    occupation codes, which do not match the 2018-coded occupation measures.
    """
    if start < SAMPLE_START:
        raise ValueError(f"--start {start} is before {SAMPLE_START}: pre-2020 CPS records "
                         "use 2010 Census occupation codes and cannot be merged")
    y, m = (int(x) for x in start.split("-"))
    return df[(df["YEAR"] > y) | ((df["YEAR"] == y) & (df["MONTH"] >= m))]


def valid_earnings(earnweek: pd.Series, var: str) -> pd.Series:
    """True for positive weekly earnings that are not the variable's NIU code."""
    niu = EARNINGS_NIU[var]
    return (earnweek > 0) & ~np.isclose(earnweek, niu) & (earnweek < niu)


def knowledge_occ_codes(occ: pd.DataFrame, majors: set = KNOWLEDGE_MAJOR) -> set:
    """CPS occupation codes whose SOC major group is in `majors`."""
    if "soc_major" not in occ.columns:
        raise ValueError("occupation codes lack soc_major; re-run step 03")
    major = occ["soc_major"].astype(str).str.zfill(2)
    return set(occ.loc[major.isin(majors), "cps_occ"])


def plain_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable integer/boolean columns (as ipumspy returns them)
    to numpy dtypes: int64 when complete, float64 (NaN) when values are missing.
    Formula-based estimation (patsy) cannot read the nullable types."""
    out = df.copy()
    for c in out.columns:
        dt = out[c].dtype
        if isinstance(dt, (pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype,
                           pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype,
                           pd.BooleanDtype)):
            out[c] = (out[c].astype("int64") if not out[c].isna().any()
                      else out[c].astype("float64"))
        elif isinstance(dt, pd.Float64Dtype) or isinstance(dt, pd.Float32Dtype):
            out[c] = out[c].astype("float64")
    return out


def prefilter(chunk: pd.DataFrame, start: str, occ_codes: set | None = None
              ) -> tuple[pd.DataFrame, list]:
    """Filters applied to each chunk as it is read, with row counts.

    Order: sample period -> basic monthly only (no ASEC) -> wage/salary ->
    employed -> knowledge-intensive occupation (when `occ_codes` is given).
    All rotation groups are kept (RQ2 uses them); the ORG earner universe is
    selected later.
    """
    chunk = plain_dtypes(chunk)
    counts = [len(chunk)]
    chunk = restrict_period(chunk, start)
    counts.append(len(chunk))
    if "ASECFLAG" in chunk.columns:
        # ASECFLAG is only set in March samples (1 = ASEC, 2 = March basic) and
        # missing in every other month; missing must be KEPT, not dropped.
        chunk = chunk[~chunk["ASECFLAG"].eq(1).fillna(False).astype(bool)]
    counts.append(len(chunk))
    if "CLASSWKR" in chunk.columns:
        chunk = chunk[chunk["CLASSWKR"].between(20, 28)]   # wage/salary, excl. self-employed
    counts.append(len(chunk))
    if "EMPSTAT" in chunk.columns:
        chunk = chunk[chunk["EMPSTAT"].isin([10, 12])]     # at work / has job, not at work
    counts.append(len(chunk))
    if occ_codes is not None:
        chunk = chunk[chunk["OCC"].isin(occ_codes)]
    counts.append(len(chunk))
    return chunk, counts


def load_ipums(ddir: pathlib.Path, start: str = SAMPLE_START, occ_codes: set | None = None,
               chunksize: int = 500_000):
    """Read the extract in chunks, applying `prefilter`; return data, summed
    counts and the DDI codebook."""
    try:
        from ipumspy import readers
    except ImportError:
        print("[sample] pip install ipumspy", file=sys.stderr)
        raise
    ddis = sorted(ddir.glob("*.xml"))
    if not ddis:
        raise FileNotFoundError(f"no DDI codebook (.xml) in {ddir}; run step 04 first")
    if len(ddis) > 1:
        raise RuntimeError(f"more than one extract codebook in {ddir}: "
                           f"{[p.name for p in ddis]}; keep only the extract to use")
    ddi = readers.read_ipums_ddi(ddis[0])
    print(f"[sample] reading {ddi.file_description.filename} in chunks of {chunksize:,}")
    parts, totals = [], None
    for chunk in readers.read_microdata_chunked(ddi, ddir / ddi.file_description.filename,
                                                chunksize=chunksize):
        chunk.columns = [c.upper() for c in chunk.columns]
        chunk, counts = prefilter(chunk, start, occ_codes)
        totals = counts if totals is None else [a + b for a, b in zip(totals, counts)]
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True), totals, ddi


def describe_variables(ddi, names: list) -> None:
    """Print the codebook description and value labels of `names` (for the log:
    the methods chapter quotes the DDI, not memory)."""
    for n in names:
        try:
            info = ddi.get_variable_info(n)
        except Exception:  # noqa: BLE001
            continue
        desc = " ".join(str(getattr(info, "description", "") or "").split())
        print(f"[ddi] {n}: {getattr(info, 'label', '')}")
        if desc:
            print(f"[ddi]   {desc[:1500]}")
        codes = getattr(info, "codes", None) or {}
        if codes:
            print(f"[ddi]   codes: {dict(list(codes.items())[:20])}")


def allocation_flags(df: pd.DataFrame, earnings_var: str) -> list:
    """Data-quality flag columns of the earnings variables (IPUMS names them Q +
    the variable name, cut to 8 characters, e.g. QEARNWEE)."""
    cands = {"Q" + earnings_var, ("Q" + earnings_var)[:8], "QEARNWEE", "QEARNWEEK",
             "QEARNWEEK2", "QEARNWE2"}
    return sorted(c for c in df.columns if c in cands)


def real_topcode(cpi: pd.DataFrame, base: float) -> float:
    """The lowest real (2019-dollar) value of the nominal top-code over the
    sample months: one cap that binds in every month."""
    return float(TOPCODE_NOMINAL * base / cpi["cpi"].max())


def parse_bands(text: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """'22-30,35-55' -> ((22, 30), (35, 55))."""
    (e0, e1), (x0, x1) = (tuple(int(v) for v in part.split("-")) for part in text.split(","))
    return (e0, e1), (x0, x1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipums", default="data/raw/ipums")
    ap.add_argument("--occ", default="data/interim/occ_measures.csv")
    ap.add_argument("--codes", default="data/interim/census_occ_codes.csv",
                    help="every Census code with its SOC major group (step 03)")
    ap.add_argument("--cpi", default="data/raw/cpi_u.csv",
                    help="columns: year,month,cpi  (BLS series CUUR0000SA0)")
    ap.add_argument("--out", default="data/out/analysis_sample.parquet")
    ap.add_argument("--employment-out", default=None,
                    help="RQ2 sample (default: <out stem>_employment.parquet)")
    ap.add_argument("--majors", default=",".join(sorted(KNOWLEDGE_MAJOR)),
                    help="knowledge-intensive SOC major groups, comma-separated")
    ap.add_argument("--bands", default="22-30,35-55",
                    help="early-career and experienced age bands")
    ap.add_argument("--keep-allocated", action="store_true")
    ap.add_argument("--include-midband", action="store_true")
    ap.add_argument("--hours", choices=["fulltime", "all"], default="fulltime",
                    help="fulltime: usual hours 35+, weekly earnings; all: hourly earnings")
    ap.add_argument("--drop-topcoded", action="store_true")
    ap.add_argument("--start", default=SAMPLE_START,
                    help="first month YYYY-MM (not earlier than 2020-01)")
    args = ap.parse_args()

    if args.start < SAMPLE_START:
        restrict_period(pd.DataFrame({"YEAR": [], "MONTH": []}), args.start)  # raises
    majors = {m.strip().zfill(2) for m in args.majors.split(",") if m.strip()}
    (e0, e1), (x0, x1) = parse_bands(args.bands)
    codes_path = pathlib.Path(args.codes)
    occ = pd.read_csv(args.occ, dtype={"soc_major": str})
    universe = pd.read_csv(codes_path, dtype={"soc_major": str}) if codes_path.exists() else occ
    if not codes_path.exists():
        print(f"[sample] WARNING: {codes_path} missing (re-run step 03); the knowledge "
              "universe is taken from codes WITH measures only")
    know = knowledge_occ_codes(universe, majors)
    print(f"[sample] SOC major groups {sorted(majors)}: {len(know)} Census codes")

    df, counts, ddi = load_ipums(pathlib.Path(args.ipums), args.start, know)
    labels = ["raw person-month records", f"{args.start} onwards (2018 Census occupation codes)",
              "basic monthly records (no ASEC)", "wage/salary workers", "employed",
              "knowledge-intensive SOC groups"]
    log = list(zip(labels, counts))
    describe_variables(ddi, ["EARNWEEK2", "EARNWEEK", HOURS_VAR, "WTFINL", "EARNWT"]
                       + [c for c in df.columns if c.startswith("Q")])

    # --- seniority bands -----------------------------------------------------
    df["early_career"] = df["AGE"].between(e0, e1).astype(int)
    df["experienced"] = df["AGE"].between(x0, x1).astype(int)
    df["potential_exp"] = (df["AGE"] - df["EDUC"].map(_educ_years) - 6).clip(lower=0)
    if not args.include_midband:
        df = df[(df["early_career"] == 1) | (df["experienced"] == 1)]
        log.append((f"in {e0}-{e1} or {x0}-{x1} age bands", len(df)))
    else:
        df = df[df["AGE"].between(e0, x1)]
        log.append((f"aged {e0}-{x1}", len(df)))

    # --- time, regime flags, treatment --------------------------------------
    df["date"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MONTH"], day=1))
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df["post"] = (df["date"] >= POST_START).astype(int)
    df["pandemic_window"] = df["date"].between("2020-04-01", "2021-06-30").astype(int)

    # --- occupation measures (records in knowledge codes without them are reported)
    no_meas = df[~df["OCC"].isin(set(occ.loc[occ["exposure"].notna(), "cps_occ"]))]
    if len(no_meas):
        top = no_meas["OCC"].value_counts()
        print(f"[sample] {len(no_meas):,} records in {top.size} knowledge-intensive Census "
              f"codes without task or exposure measures (dropped): {top.to_dict()}")
    df = df.merge(occ, left_on="OCC", right_on="cps_occ", how="inner")
    df = df.dropna(subset=["z3", "exposure"])
    log.append(("with task and exposure measures (RQ2 employment sample)", len(df)))

    emp_out = pathlib.Path(args.employment_out or
                           pathlib.Path(args.out).with_name(pathlib.Path(args.out).stem
                                                            + "_employment.parquet"))
    emp_cols = ["YEAR", "MONTH", "OCC", "STATEFIP", "quarter", "post", "pandemic_window",
                "early_career", "experienced", "exposure", "WTFINL", "SEX", "AGE", "EDUC"]
    if "WTFINL" in df.columns:
        emp_out.parent.mkdir(parents=True, exist_ok=True)
        df[[c for c in emp_cols if c in df.columns]].to_parquet(emp_out, index=False)
    else:
        print("[sample] WARNING: WTFINL not in the extract; no RQ2 employment sample written "
              "(step 07 falls back to the earnings sample)")

    # --- ORG earner universe and earnings -----------------------------------
    df = df[df["EARNWT"] > 0]
    log.append(("in ORG earner universe", len(df)))
    ew = "EARNWEEK2" if "EARNWEEK2" in df.columns else "EARNWEEK"
    if ew == "EARNWEEK":
        print("[sample] WARNING: EARNWEEK2 absent; pre/post-2023 earnings are NOT "
              "comparable without it (Census rounding introduced Apr 2023).")
    df["earnweek"] = pd.to_numeric(df[ew], errors="coerce")
    df = df[valid_earnings(df["earnweek"], ew)]
    log.append(("valid weekly earnings", len(df)))

    flags = allocation_flags(df, ew)
    if flags:
        allocated = (df[flags].fillna(0) != 0).any(axis=1)
        for f in flags:
            print(f"[sample] {f} values: {df[f].value_counts(dropna=False).sort_index().to_dict()}")
        share = allocated.groupby(df["quarter"]).mean()
        print("[sample] share of earnings allocated (imputed) by quarter: "
              + ", ".join(f"{q} {v:.1%}" for q, v in share.items()))
        df["allocated"] = allocated.astype(int)
        if not args.keep_allocated:
            df = df[~allocated]
            log.append(("non-allocated earnings (Hirsch-Schumacher)", len(df)))
    else:
        print("[sample] WARNING: no earnings allocation flag in the extract (request it "
              "with data quality flags, step 04); imputed earnings are NOT excluded")

    hvar = HOURS_VAR if HOURS_VAR in df.columns else None
    if hvar is None:
        print(f"[sample] WARNING: {HOURS_VAR} not in the extract; no hours restriction")
    else:
        h = pd.to_numeric(df[hvar], errors="coerce")
        print(f"[sample] {hvar}: {h.between(1, HOURS_MAX).mean():.1%} report usual hours, "
              f"{(h == 997).mean():.1%} 'hours vary', {h.between(35, HOURS_MAX).mean():.1%} "
              "full-time (35+)")
        if args.hours == "fulltime":
            df = df[h.between(35, HOURS_MAX)]
            log.append(("full-time (usual hours 35+)", len(df)))
        else:
            df = df[h.between(1, HOURS_MAX)]
            log.append(("usual hours reported", len(df)))

    cpi = pd.read_csv(args.cpi)
    cpi.columns = [c.lower() for c in cpi.columns]
    base = cpi[cpi["year"] == 2019]["cpi"].mean()
    df = df.merge(cpi.rename(columns={"year": "YEAR", "month": "MONTH"}),
                  on=["YEAR", "MONTH"], how="left")
    if df["cpi"].isna().any():
        print(f"[sample] WARNING: {df['cpi'].isna().sum()} rows lack a CPI match")
    months = cpi.merge(df[["YEAR", "MONTH"]].drop_duplicates()
                       .rename(columns={"YEAR": "year", "MONTH": "month"}))
    cap = real_topcode(months, base)
    df["real_earnweek"] = df["earnweek"] * base / df["cpi"]
    df["is_topcoded"] = (df["real_earnweek"] >= cap - 1e-9).astype(int)
    share = df.groupby("quarter")["is_topcoded"].mean()
    print(f"[sample] common real top-code ${cap:,.2f} (2019 dollars); share at or above it "
          "by quarter: " + ", ".join(f"{q} {v:.1%}" for q, v in share.items()))
    if args.drop_topcoded:
        df = df[df["is_topcoded"] == 0]
        log.append(("below the common real top-code", len(df)))
    df["real_earnweek"] = df["real_earnweek"].clip(upper=cap)
    if args.hours == "all" and hvar:
        df["ln_w"] = np.log(df["real_earnweek"] / pd.to_numeric(df[hvar]))
        print("[sample] outcome ln_w = log real HOURLY earnings (weekly / usual hours)")
    else:
        df["ln_w"] = np.log(df["real_earnweek"])
    log.append(("earnings sample (RQ1, RQ3, RQ4)", len(df)))

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print("\n[sample] SAMPLE CONSTRUCTION LOG (paste into Table 4.1; person-month records):")
    for label, n in log:
        print(f"    {label:<56} {n:>10,}")
    print(f"\n[sample] early-career: {int(df['early_career'].sum()):,} | "
          f"experienced: {int(df['experienced'].sum()):,}")
    print(f"[sample] written to {args.out}" + (f" and {emp_out}" if "WTFINL" in df else ""))
    return 0


def _educ_years(code: float) -> float:
    """Approximate years of schooling from IPUMS CPS EDUC codes."""
    try:
        c = int(code)
    except (TypeError, ValueError):
        return np.nan
    table = {2: 0, 10: 2, 20: 5.5, 30: 7.5, 40: 9, 50: 10, 60: 11, 70: 12,
             71: 12, 73: 12, 80: 13, 81: 13, 90: 14, 91: 14, 92: 14, 100: 15,
             110: 16, 111: 16, 120: 17, 121: 17, 122: 18, 123: 18, 124: 19, 125: 21}
    return table.get(c, np.nan)


if __name__ == "__main__":
    raise SystemExit(main())
