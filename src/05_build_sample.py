"""
05_build_sample.py — construct the estimation sample from the IPUMS extract.

Implements every cleaning rule the proposal commits to in Sections 6.2 and 6.6:
  * ORG earnings universe only (wage/salary workers, not self-employed)
  * knowledge-intensive SOC major groups 13,15,17,19,23,27,43 (major group of
    each CPS occupation code from the Census crosswalk, step 03)
  * early-career (22-30) vs experienced (35-55); 31-34 excluded from the
    primary contrast, restorable via --include-midband
  * harmonised ROUNDED weekly earnings (EARNWEEK2) so pre/post-2023 are
    comparable after the Census privacy-protection changes
  * allocated (imputed) earnings EXCLUDED by default — Hirsch & Schumacher (2004)
    match bias — restorable via --keep-allocated
  * sample period January 2020 onwards: CPS switched to 2018 Census occupation
    codes in January 2020, and the occupation measures are keyed on those codes,
    so earlier records would merge to the wrong occupations
  * CPI-U deflation to 2019 dollars
  * top-code flagging so the dynamic top-code from Apr 2023 can be trimmed

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


def knowledge_occ_codes(occ: pd.DataFrame) -> set:
    """CPS occupation codes whose SOC major group is knowledge-intensive."""
    if "soc_major" not in occ.columns:
        raise ValueError("occupation measures lack soc_major; re-run step 03")
    major = occ["soc_major"].astype(str).str.zfill(2)
    return set(occ.loc[major.isin(KNOWLEDGE_MAJOR), "cps_occ"])


def prefilter(chunk: pd.DataFrame, start: str) -> tuple[pd.DataFrame, list]:
    """Filters that can be applied to each chunk as it is read, with row counts.

    Order: sample period -> basic monthly only (no ASEC) -> ORG earner universe.
    Applying them per chunk keeps memory to the ~1/4 of records in the earner
    study instead of the full ~9 million-record file.
    """
    counts = [len(chunk)]
    chunk = restrict_period(chunk, start)
    counts.append(len(chunk))
    if "ASECFLAG" in chunk.columns:
        chunk = chunk[chunk["ASECFLAG"] != 1]
    counts.append(len(chunk))
    chunk = chunk[chunk["EARNWT"] > 0]
    counts.append(len(chunk))
    return chunk, counts


def load_ipums(ddir: pathlib.Path, start: str = SAMPLE_START,
               chunksize: int = 500_000) -> tuple[pd.DataFrame, list]:
    """Read the extract in chunks, applying `prefilter`; return data and summed counts."""
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
    parts, totals = [], [0, 0, 0, 0]
    for chunk in readers.read_microdata_chunked(ddi, ddir / ddi.file_description.filename,
                                                chunksize=chunksize):
        chunk.columns = [c.upper() for c in chunk.columns]
        chunk, counts = prefilter(chunk, start)
        totals = [a + b for a, b in zip(totals, counts)]
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True), totals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipums", default="data/raw/ipums")
    ap.add_argument("--occ", default="data/interim/occ_measures.csv")
    ap.add_argument("--cpi", default="data/raw/cpi_u.csv",
                    help="columns: year,month,cpi  (BLS series CUUR0000SA0)")
    ap.add_argument("--out", default="data/out/analysis_sample.parquet")
    ap.add_argument("--keep-allocated", action="store_true")
    ap.add_argument("--include-midband", action="store_true")
    ap.add_argument("--start", default=SAMPLE_START,
                    help="first month YYYY-MM (not earlier than 2020-01)")
    args = ap.parse_args()

    if args.start < SAMPLE_START:
        restrict_period(pd.DataFrame({"YEAR": [], "MONTH": []}), args.start)  # raises
    df, (n_raw, n_period, n_basic, n_org) = load_ipums(pathlib.Path(args.ipums), args.start)
    log = [("raw records", n_raw),
           (f"{args.start} onwards (2018 Census occupation codes)", n_period),
           ("basic monthly records (no ASEC)", n_basic),
           ("in ORG earner universe", n_org)]

    # --- wage/salary, employed -----------------------------------------------
    df = df[df["CLASSWKR"].between(20, 28)]          # wage/salary, excl. self-employed
    log.append(("wage/salary workers", len(df)))
    df = df[df["EMPSTAT"].isin([10, 12])]            # employed at work / has job
    log.append(("employed", len(df)))

    # --- occupation restriction (SOC major group from the step 03 crosswalk) --
    occ = pd.read_csv(args.occ, dtype={"soc_major": str})
    df = df[df["OCC"].isin(knowledge_occ_codes(occ))]
    log.append(("knowledge-intensive SOC groups", len(df)))

    # --- seniority bands -----------------------------------------------------
    df["early_career"] = df["AGE"].between(22, 30).astype(int)
    df["experienced"] = df["AGE"].between(35, 55).astype(int)
    df["potential_exp"] = (df["AGE"] - df["EDUC"].map(_educ_years) - 6).clip(lower=0)
    if not args.include_midband:
        df = df[(df["early_career"] == 1) | (df["experienced"] == 1)]
        log.append(("in 22-30 or 35-55 age bands", len(df)))

    # --- earnings ------------------------------------------------------------
    ew = "EARNWEEK2" if "EARNWEEK2" in df.columns else "EARNWEEK"
    if ew == "EARNWEEK":
        print("[sample] WARNING: EARNWEEK2 absent; pre/post-2023 earnings are NOT "
              "comparable without it (Census rounding introduced Apr 2023).")
    df["earnweek"] = pd.to_numeric(df[ew], errors="coerce")
    df = df[valid_earnings(df["earnweek"], ew)]
    log.append(("valid weekly earnings", len(df)))

    if not args.keep_allocated:
        flag = next((c for c in ("QEARNWEE", "QEARNWEEK") if c in df.columns), None)
        if flag:
            df = df[df[flag] == 0]
            log.append(("non-allocated earnings (Hirsch-Schumacher)", len(df)))
        else:
            print("[sample] WARNING: no EARNWEEK allocation flag in the extract; "
                  "imputed earnings are NOT excluded")

    # dynamic top-code flag: mark the monthly maximum, which is the top-coded value
    df["is_topcoded"] = (df.groupby(["YEAR", "MONTH"])["earnweek"]
                           .transform("max") == df["earnweek"]).astype(int)

    cpi = pd.read_csv(args.cpi)
    cpi.columns = [c.lower() for c in cpi.columns]
    base = cpi[cpi["year"] == 2019]["cpi"].mean()
    df = df.merge(cpi.rename(columns={"year": "YEAR", "month": "MONTH"}),
                  on=["YEAR", "MONTH"], how="left")
    if df["cpi"].isna().any():
        print(f"[sample] WARNING: {df['cpi'].isna().sum()} rows lack a CPI match")
    df["real_earnweek"] = df["earnweek"] * base / df["cpi"]
    df["ln_w"] = np.log(df["real_earnweek"])

    # --- time, regime flags, treatment --------------------------------------
    df["date"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MONTH"], day=1))
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df["post"] = (df["date"] >= "2022-12-01").astype(int)
    df["pandemic_window"] = df["date"].between("2020-04-01", "2021-06-30").astype(int)

    # --- merge occupation-level task + exposure measures ---------------------
    df = df.merge(occ, left_on="OCC", right_on="cps_occ", how="left")
    miss = df["z3"].isna().mean()
    print(f"[sample] {miss:.1%} of person-records lack occupation measures after merge")
    df = df.dropna(subset=["z3", "exposure"])
    log.append(("merged to task/exposure measures", len(df)))

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print("\n[sample] SAMPLE CONSTRUCTION LOG (paste into Table 4.1):")
    for label, n in log:
        print(f"    {label:<48} {n:>10,}")
    print(f"\n[sample] early-career: {int(df['early_career'].sum()):,} | "
          f"experienced: {int(df['experienced'].sum()):,}")
    print(f"[sample] written to {args.out}")
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
