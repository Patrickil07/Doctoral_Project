"""
03_crosswalk.py — map occupation-level task composition and GenAI exposure
onto CPS occupation codes, weighting by OEWS employment.

FIXES in this version
  1. Column detection: the Census file has title rows above the real header,
     and one title mentions "census", "SOC" and "code" in a single cell. The
     old search picked that cell for BOTH columns, producing two columns named
     soc2018 -> AttributeError: 'DataFrame' object has no attribute 'str'.
     Now the two columns must be distinct, each matching one concept only.
  2. Wildcard SOC codes (e.g. 13-20XX) are expanded to every matching detailed
     SOC in the task data. The old pattern accepted only all-digit codes and
     silently dropped them,
     removing aggregated occupations from the crosswalk.
  3. Manual override: if detection still fails, set OCC_COL / SOC_COL below to
     the exact header text in your file.

Inputs (all free, all public):
  data/raw/oews_national.xlsx          OEWS national, May 2021 (first year fully on SOC 2018)
  data/raw/census_soc_crosswalk.xlsx   Census 2018 occupation code list w/ crosswalk
  data/interim/exposure_soc2018.csv    columns: soc2018,exposure (step 00b)
"""
import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd

PARTS = ["c1_nonroutine_analytic", "c2_nonroutine_interpersonal",
         "c3_routine_cognitive", "c4_residual"]

# Leave as None for automatic detection. If it fails, open the crosswalk file
# and paste the exact header text, e.g. OCC_COL = "2018 Census Code".
OCC_COL = None
SOC_COL = None


def load_oews(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)
    df.columns = [str(c).strip().upper() for c in df.columns]
    soc_col = next((c for c in df.columns if c in ("OCC_CODE", "OCC CODE")), None)
    emp_col = next((c for c in df.columns if c.startswith("TOT_EMP")), None)
    if not soc_col or not emp_col:
        raise ValueError(f"OEWS schema unexpected; columns seen: {list(df.columns)[:15]}")
    out = df[[soc_col, emp_col]].copy()
    out.columns = ["soc2018", "emp"]
    out["emp"] = pd.to_numeric(out["emp"].astype(str).str.replace(r"[^\d.]", "", regex=True),
                               errors="coerce")
    out = out.dropna(subset=["emp"])
    return out[out["soc2018"].str.match(r"^\d{2}-\d{4}$", na=False)]


def load_crosswalk(path: pathlib.Path, occ_col: str | None = None,
                   soc_col: str | None = None) -> pd.DataFrame:
    """Census occupation code <-> SOC 2018.

    The Census file has title rows above the real header, and some of those
    titles mention census, SOC *and* code in one cell. So we require two
    DIFFERENT columns, each matching one concept and not the other, and we
    confirm the choice by checking the values actually look like codes.
    """
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        n = len(xl.parse(sheet, header=None))
        for hdr in range(0, min(15, n)):
            df = xl.parse(sheet, dtype=str, header=hdr)
            cols = {str(c).strip().lower(): c for c in df.columns}
            if occ_col and soc_col:
                oc = next((v for k, v in cols.items() if k == occ_col.lower()), None)
                sc = next((v for k, v in cols.items() if k == soc_col.lower()), None)
            else:
                oc = next((v for k, v in cols.items()
                           if "census" in k and "code" in k and "soc" not in k), None)
                sc = next((v for k, v in cols.items()
                           if "soc" in k and "code" in k and "census" not in k), None)
            if oc is None or sc is None or oc == sc:
                continue
            out = df[[oc, sc]].copy()
            out.columns = ["cps_occ", "soc_pattern"]          # set, never rename-collide
            out = out.dropna()
            # keep real occupation rows only: section headers carry code ranges
            # such as "0010-0440" / "11-0000 - 13-0000" and are not occupations
            out = out[out["cps_occ"].astype(str).str.strip().str.fullmatch(r"\d{4}")]
            out["soc_pattern"] = (out["soc_pattern"].astype(str).str.strip().str.upper()
                                  .str.extract(r"(\d{2}-[\dX]{4})")[0])
            out["cps_occ"] = pd.to_numeric(
                out["cps_occ"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
            out = out.dropna()
            if len(out) >= 5:                                  # values really are codes
                n_wild = out["soc_pattern"].str.contains("X").sum()
                print(f"[xwalk] sheet '{sheet}', header row {hdr}: "
                      f"'{oc}' -> '{sc}' ({len(out)} pairs, {n_wild} wildcard SOC codes)")
                out["cps_occ"] = out["cps_occ"].astype(int)
                return out
    raise ValueError(
        "could not locate distinct Census-code and SOC-code columns. Open the file, "
        "note the exact header names, and pass --occ-col and --soc-col.")


def expand_wildcards(xw: pd.DataFrame, known_socs: pd.Series) -> pd.DataFrame:
    """Expand non-detailed SOC codes to the detailed SOCs in the task data.

    '13-20XX' -> every known SOC starting '13-20'. A broad-group code ending in
    0 that is not itself a detailed SOC (e.g. Census '15-1230' for 15-1231 and
    15-1232) is treated the same way, as '15-123X'.
    """
    known = sorted(set(known_socs.dropna().astype(str)))
    known_set = set(known)
    rows, unmatched = [], []
    for occ, pat in xw[["cps_occ", "soc_pattern"]].itertuples(index=False):
        if pat.endswith("0") and pat not in known_set and "X" not in pat:
            pat = pat[:-1] + "X"
        if "X" not in pat:
            rows.append((occ, pat)); continue
        rx = re.compile("^" + pat.replace("X", r"\d") + "$")
        hits = [s for s in known if rx.match(s)]
        if hits:
            rows.extend((occ, s) for s in hits)
        else:
            unmatched.append(pat)
    if unmatched:
        print(f"[xwalk] {len(unmatched)} wildcard codes matched no task-data SOC: "
              f"{sorted(set(unmatched))[:8]}")
    out = pd.DataFrame(rows, columns=["cps_occ", "soc2018"]).drop_duplicates()
    print(f"[xwalk] {len(xw)} Census rows -> {len(out)} Census x detailed-SOC pairs "
          f"after wildcard expansion")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/interim/task_composition.csv")
    ap.add_argument("--oews", default="data/raw/oews_national.xlsx")
    ap.add_argument("--crosswalk", default="data/raw/census_soc_crosswalk.xlsx")
    ap.add_argument("--exposure", default="data/interim/exposure_soc2018.csv")
    ap.add_argument("--out", default="data/interim/occ_measures.csv")
    ap.add_argument("--occ-col", default=OCC_COL)
    ap.add_argument("--soc-col", default=SOC_COL)
    args = ap.parse_args()

    for p in (args.tasks, args.oews, args.crosswalk, args.exposure):
        if not pathlib.Path(p).exists():
            print(f"[xwalk] missing input: {p}\n"
                  "        See the docstring for where to download it.", file=sys.stderr)
            return 1

    tasks = pd.read_csv(args.tasks)
    # collapse O*NET-SOC detail (e.g. 15-1252.01) to detailed SOC (15-1252)
    tasks = tasks.groupby("soc2018")[PARTS + ["ln_T"]].mean().reset_index()

    exposure = pd.read_csv(args.exposure, dtype={"soc2018": str})
    if "exposure" not in exposure.columns:
        print("[xwalk] exposure file must have columns: soc2018,exposure", file=sys.stderr)
        return 1

    oews = load_oews(pathlib.Path(args.oews))
    xw_raw = load_crosswalk(pathlib.Path(args.crosswalk), args.occ_col, args.soc_col)
    xw = expand_wildcards(xw_raw, tasks["soc2018"])

    df = (xw.merge(tasks, on="soc2018", how="left")
            .merge(exposure[["soc2018", "exposure"]], on="soc2018", how="left")
            .merge(oews, on="soc2018", how="left"))

    no_task = df["c1_nonroutine_analytic"].isna().mean()
    no_exp = df["exposure"].isna().mean()
    no_emp = df["emp"].isna().mean()
    print(f"[xwalk] pairs lacking task data {no_task:.1%} | exposure {no_exp:.1%} | "
          f"OEWS employment {no_emp:.1%}")
    df = df.dropna(subset=PARTS + ["exposure"])
    if df.empty:
        print("[xwalk] nothing left after merging - check that SOC codes in the three "
              "inputs use the same format (e.g. 15-1252)", file=sys.stderr)
        return 1
    df["emp"] = df["emp"].fillna(df["emp"].median())

    # employment-weighted aggregation to CPS occupation level
    def wavg(g: pd.DataFrame) -> pd.Series:
        w = g["emp"].to_numpy(dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.repeat(1 / len(g), len(g))
        vals = {c: float(np.dot(w, g[c].to_numpy(dtype=float)))
                for c in PARTS + ["ln_T", "exposure"]}
        vals["n_soc"] = len(g)
        vals["emp_total"] = float(g["emp"].sum())
        return pd.Series(vals)

    occ = df.groupby("cps_occ").apply(wavg, include_groups=False).reset_index()

    # re-close composition after weighted averaging, then recompute ILR
    C = occ[PARTS].to_numpy()
    C = C / C.sum(axis=1, keepdims=True)
    c1, c2, c3, c4 = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
    occ[PARTS] = C
    occ["z1"] = np.sqrt(3 / 4) * np.log(np.cbrt(c1 * c2 * c3) / c4)
    occ["z2"] = np.sqrt(2 / 3) * np.log(np.sqrt(c1 * c3) / c2)
    occ["z3"] = np.sqrt(1 / 2) * np.log(c1 / c3)

    # exposure terciles for the robustness specification
    occ["exposure_tercile"] = pd.qcut(occ["exposure"], 3, labels=["low", "mid", "high"],
                                      duplicates="drop")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    occ.to_csv(args.out, index=False)
    print(f"[xwalk] {len(occ)} CPS occupation codes written to {args.out}")
    print(occ[["z3", "exposure"]].describe().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
