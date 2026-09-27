"""
05_build_sample.py — construct the estimation sample from the IPUMS extract.

Implements every cleaning rule the proposal commits to in Sections 6.2 and 6.6:
  * ORG earnings universe only (wage/salary workers, not self-employed)
  * knowledge-intensive SOC major groups 13,15,17,19,23,27,43
  * early-career (22-30) vs experienced (35-55); 31-34 excluded from the
    primary contrast, restorable via --include-midband
  * harmonised ROUNDED weekly earnings (EARNWEEK2) so pre/post-2023 are
    comparable after the Census privacy-protection changes
  * allocated (imputed) earnings EXCLUDED by default — Hirsch & Schumacher (2004)
    match bias — restorable via --keep-allocated
  * CPI-U deflation to 2019 dollars
  * top-code flagging so the dynamic top-code from Apr 2023 can be trimmed
  * occupation-coding regime flag for the January 2020 break

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


def load_ipums(ddir: pathlib.Path) -> pd.DataFrame:
    try:
        from ipumspy import readers
    except ImportError:
        print("[sample] pip install ipumspy", file=sys.stderr)
        raise
    ddis = list(ddir.glob("*.xml"))
    if not ddis:
        raise FileNotFoundError(f"no DDI codebook (.xml) in {ddir}; run step 04 first")
    ddi = readers.read_ipums_ddi(ddis[0])
    df = readers.read_microdata(ddi, ddir / ddi.file_description.filename)
    df.columns = [c.upper() for c in df.columns]
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipums", default="data/raw/ipums")
    ap.add_argument("--occ", default="data/interim/occ_measures.csv")
    ap.add_argument("--cpi", default="data/raw/cpi_u.csv",
                    help="columns: year,month,cpi  (BLS series CUUR0000SA0)")
    ap.add_argument("--out", default="data/out/analysis_sample.parquet")
    ap.add_argument("--keep-allocated", action="store_true")
    ap.add_argument("--include-midband", action="store_true")
    args = ap.parse_args()

    df = load_ipums(pathlib.Path(args.ipums))
    n0 = len(df)
    log = [("raw records", n0)]

    # --- ORG earner universe -------------------------------------------------
    df = df[df["EARNWT"] > 0]
    log.append(("in ORG earner universe", len(df)))
    df = df[df["CLASSWKR"].between(20, 28)]          # wage/salary, excl. self-employed
    log.append(("wage/salary workers", len(df)))
    df = df[df["EMPSTAT"].isin([10, 12])]            # employed at work / has job
    log.append(("employed", len(df)))

    # --- occupation restriction ---------------------------------------------
    df["OCCSOC"] = df["OCCSOC"].astype(str).str.strip()
    df["soc_major"] = df["OCCSOC"].str.slice(0, 2)
    df = df[df["soc_major"].isin(KNOWLEDGE_MAJOR)]
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
    df = df[(df["earnweek"] > 0) & (df["earnweek"] < 99999)]
    log.append(("valid weekly earnings", len(df)))

    if not args.keep_allocated:
        if "QEARNWEEK" in df.columns:
            df = df[df["QEARNWEEK"] == 0]
            log.append(("non-allocated earnings (Hirsch-Schumacher)", len(df)))
        else:
            print("[sample] WARNING: QEARNWEEK missing; cannot exclude imputed earnings")

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
    df["occ_code_regime"] = np.where(df["date"] < "2020-01-01", "census2010", "census2018")
    df["pandemic_window"] = df["date"].between("2020-04-01", "2021-06-30").astype(int)

    # --- merge occupation-level task + exposure measures ---------------------
    occ = pd.read_csv(args.occ)
    pre2020 = (df["occ_code_regime"] == "census2010").mean()
    if pre2020 > 0:
        print(f"[sample] WARNING: {pre2020:.1%} of records predate January 2020 and carry "
              "2010 Census occupation codes,\n"
              "         but occupation measures are keyed on 2018 Census codes. Codes that "
              "changed in 2018 will\n"
              "         merge to the wrong occupation or not at all. Recode them with the "
              "Census 2010->2018\n"
              "         crosswalk, or restrict to census2018 (see data/README.md).")
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
