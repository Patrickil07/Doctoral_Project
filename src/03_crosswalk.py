"""
03_crosswalk.py — map occupation-level task composition and GenAI exposure
onto CPS occupation codes, weighting by OEWS employment.

FIXES in this version
  1. Column detection: the Census file has title rows above the real header,
     and one title mentions "census", "SOC" and "code" in a single cell. The
     old search picked that cell for BOTH columns, producing two columns named
     soc2018 -> AttributeError: 'DataFrame' object has no attribute 'str'.
     Now the two columns must be distinct, each matching one concept only.
  2. Wildcard SOC codes (e.g. 13-20XX) are expanded to the matching detailed
     SOCs in the task data that no other Census row names (see
     expand_wildcards; giving them every matching SOC double-counted the
     separately coded ones).
  3. Manual override: if detection still fails, set OCC_COL / SOC_COL below to
     the exact header text in your file.

Inputs (all free, all public):
  data/raw/oews_national.xlsx          OEWS national, May 2021 (first year fully on SOC 2018)
  data/raw/census_soc_crosswalk.xlsx   Census 2018 occupation code list w/ crosswalk
  data/interim/exposure_soc2018.csv    columns: soc2018,exposure (step 00b;
                                       primary = Eloundou et al. human-rated beta)
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
    out = out[out["soc2018"].str.match(r"^\d{2}-\d{4}$", na=False)]
    # A code can appear on more than one row (e.g. listed at two aggregation
    # levels). Keep one row per code, so merges never duplicate SOC pairs.
    dup = out[out["soc2018"].duplicated(keep=False)]
    if len(dup):
        print(f"[xwalk] OEWS lists {dup['soc2018'].nunique()} codes on several rows; "
              f"keeping the largest employment for each: "
              f"{dup.groupby('soc2018')['emp'].apply(list).to_dict()}")
    return out.groupby("soc2018", as_index=False)["emp"].max()


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


def census_pattern(pat: str, known_set: set) -> str:
    """Normalise a Census SOC entry to an exact detailed SOC or an X pattern.

    A code ending in 0 that is not itself a detailed SOC is a group code: all
    its trailing zeros become X, so broad '15-1230' -> '15-123X' and minor
    '25-1000' -> '25-1XXX' (replacing only the last zero made minor-group codes
    match nothing).
    """
    if "X" in pat or pat in known_set or not pat.endswith("0"):
        return pat
    stem = pat.rstrip("0")
    return stem + "X" * (len(pat) - len(stem))


def expand_wildcards(xw: pd.DataFrame, known_socs: pd.Series) -> pd.DataFrame:
    """Assign every detailed SOC in the task data to the Census codes that cover it.

    In the Census list, 'X' means "the detailed codes of this group that are
    NOT listed separately": '13-20XX' (Other financial specialists) covers the
    13-20xx SOCs that no other Census row names. So a SOC named explicitly by
    one Census row is never also given to a wildcard row, and among wildcard
    rows the most specific pattern (fewest X) takes it first.

    Returns cps_occ, soc2018, share: share = 1 / (number of Census codes the
    SOC is assigned to), so a SOC listed under two codes splits its employment
    instead of being counted twice. Such SOCs are printed.
    """
    known = sorted(set(known_socs.dropna().astype(str)))
    known_set = set(known)
    pats = [(occ, census_pattern(p, known_set))
            for occ, p in xw[["cps_occ", "soc_pattern"]].itertuples(index=False)]
    rows = [(occ, p) for occ, p in pats if "X" not in p]
    taken = {p for _, p in rows}
    unmatched = []
    wild = sorted(((occ, p) for occ, p in pats if "X" in p), key=lambda t: t[1].count("X"))
    for n_x in sorted({p.count("X") for _, p in wild}):
        level = [(occ, p) for occ, p in wild if p.count("X") == n_x]
        new = set()
        for occ, pat in level:
            rx = re.compile("^" + pat.replace("X", r"\d") + "$")
            hits = [s for s in known if rx.match(s) and s not in taken]
            if hits:
                rows.extend((occ, s) for s in hits)
                new.update(hits)
            else:
                unmatched.append((occ, pat))
        taken |= new
    if unmatched:
        print(f"[xwalk] {len(unmatched)} Census codes matched no task-data SOC "
              f"(no measures for them): {sorted(unmatched)}")
    out = pd.DataFrame(rows, columns=["cps_occ", "soc2018"]).drop_duplicates()
    n_codes = out.groupby("soc2018")["cps_occ"].transform("nunique")
    out["share"] = 1.0 / n_codes
    multi = out[n_codes > 1].groupby("soc2018")["cps_occ"].apply(sorted)
    if len(multi):
        print(f"[xwalk] {len(multi)} SOCs are listed under more than one Census code; "
              f"their employment is split equally: {dict(multi)}")
    print(f"[xwalk] {len(xw)} Census rows -> {len(out)} Census x detailed-SOC pairs "
          f"after wildcard expansion; {out['soc2018'].nunique()} distinct SOCs")
    return out


def fill_employment(df: pd.DataFrame, oews: pd.DataFrame) -> pd.DataFrame:
    """Employment for detailed SOCs that OEWS does not publish separately.

    The employment of the SOC's broad group (15-1250 for 15-1252), less what its
    published detailed SOCs account for, is split equally among its unpublished
    detailed SOCs; failing that, the minor group (15-1200). A SOC with neither
    gets 0 (it then counts only if its Census code has no other SOC) and is
    listed. Replaces a fill with the median of all SOCs, which invented weights.
    """
    df = df.copy()
    pub = oews.groupby("soc2018")["emp"].max()
    miss = sorted(set(df.loc[df["emp"].isna(), "soc2018"]))
    filled, none = {}, []
    detailed_pub = {s for s in pub.index if not s.endswith("0")}
    for level, group_of in (("broad", lambda s: s[:6] + "0"), ("minor", lambda s: s[:5] + "00")):
        todo = [s for s in miss if s not in filled]
        groups = {}
        for s in todo:
            groups.setdefault(group_of(s), []).append(s)
        for g, socs in groups.items():
            if g not in pub.index:
                continue
            covered = (sum(pub[d] for d in detailed_pub if group_of(d) == g)
                       + sum(v for d, (v, _) in filled.items() if group_of(d) == g))
            rest = pub[g] - covered
            if rest > 0:
                for s in socs:
                    filled[s] = (rest / len(socs), f"{level} {g}")
    for s in miss:
        if s not in filled:
            none.append(s)
            filled[s] = (0.0, "none")
    for s, (v, how) in sorted(filled.items()):
        print(f"[xwalk] OEWS employment for {s} imputed from {how}: {v:,.0f}")
    df.loc[df["emp"].isna(), "emp"] = df.loc[df["emp"].isna(), "soc2018"].map(
        lambda s: filled[s][0])
    if none:
        print(f"[xwalk] WARNING: {len(none)} SOCs have no OEWS employment at any level "
              f"(weight 0): {none}")
    return df


def collapse_onet_soc(tasks: pd.DataFrame, cols: list) -> pd.DataFrame:
    """One row per 6-digit SOC: the main O*NET-SOC occupation (.00) where it
    exists, otherwise the unweighted mean of the detail codes (step 00b applies
    the same rule to exposure)."""
    t = tasks.copy()
    t["onet_soc"] = t.get("onet_soc", t["soc2018"]).astype(str)
    main = t["onet_soc"].str.endswith(".00")
    has_main = set(t.loc[main, "soc2018"])
    t = t[main | ~t["soc2018"].isin(has_main)]
    return t.groupby("soc2018")[cols].mean().reset_index()


def weighted_terciles(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    """Upper bounds of the low and mid thirds of total weight: each occupation
    goes to the third holding the midpoint of its weight (sorted by value)."""
    d = pd.DataFrame({"v": values, "w": weights}).dropna().sort_values("v")
    share = d["w"] / d["w"].sum()
    third = np.minimum((3 * (share.cumsum() - share / 2)).astype(int), 2)
    lo = float(d.loc[third == 0, "v"].max()) if (third == 0).any() else -np.inf
    hi = float(d.loc[third <= 1, "v"].max()) if (third <= 1).any() else lo
    return lo, hi


def census_code_majors(xw: pd.DataFrame) -> pd.DataFrame:
    """SOC major group of EVERY Census occupation code in the crosswalk, with or
    without task data (step 05 selects the knowledge universe from this list, so
    codes without measures are counted, not silently dropped)."""
    d = xw.assign(soc_major=xw["soc_pattern"].str.slice(0, 2))
    return d.drop_duplicates("cps_occ")[["cps_occ", "soc_major"]].sort_values("cps_occ")


def _wavg(values: np.ndarray, emp: np.ndarray) -> float:
    """Employment-weighted mean; equal weights if employment sums to zero."""
    w = emp / emp.sum() if emp.sum() > 0 else np.repeat(1 / len(emp), len(emp))
    return float(np.dot(w, values))


# Knowledge-intensive SOC major groups (step 05 uses the same list; a test
# checks they agree). Only used here for the exposure terciles.
KNOWLEDGE_MAJOR = {"13", "15", "17", "19", "23", "27", "43"}


def aggregate_to_cps(df: pd.DataFrame) -> pd.DataFrame:
    """Employment-weighted task and exposure measures per CPS occupation code.

    Each measure is averaged over the SOC occupations that have it: task
    shares over SOC codes with task data, exposure over SOC codes with an
    exposure score. So the task measures do not depend on which exposure
    file is used, and the exposure robustness runs vary exposure only.
    """
    def one(g: pd.DataFrame) -> pd.Series:
        emp = (g["emp"] * g.get("share", 1.0)).to_numpy(dtype=float)
        vals = {c: _wavg(g[c].to_numpy(dtype=float), emp) for c in PARTS + ["ln_T"]}
        if "task_filled" in g:
            # share of the code's employment whose task measures come from the
            # --fill-from release (step 02)
            vals["task_filled_share"] = _wavg(g["task_filled"].to_numpy(dtype=float), emp)
        has = g["exposure"].notna().to_numpy()
        vals["exposure"] = (_wavg(g["exposure"].to_numpy(dtype=float)[has], emp[has])
                            if has.any() else np.nan)
        vals["n_soc"] = len(g)
        vals["n_soc_exposure"] = int(has.sum())
        vals["emp_total"] = float(emp.sum())
        return pd.Series(vals)

    return df.groupby("cps_occ").apply(one, include_groups=False).reset_index()


def soc_major_by_occ(df: pd.DataFrame) -> pd.DataFrame:
    """SOC 2018 major group (first two digits) of each CPS occupation code.

    Where a Census code spans SOC codes in more than one major group, the group
    with the most OEWS employment is used. Replaces IPUMS OCCSOC, which the CPS
    collection does not offer.
    """
    d = df.assign(soc_major=df["soc2018"].astype(str).str.slice(0, 2),
                  emp=df["emp"] * df.get("share", 1.0))
    emp = d.groupby(["cps_occ", "soc_major"])["emp"].sum().reset_index()
    emp = emp.sort_values(["cps_occ", "emp", "soc_major"], ascending=[True, False, True])
    return emp.drop_duplicates("cps_occ")[["cps_occ", "soc_major"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/interim/task_composition.csv")
    ap.add_argument("--oews", default="data/raw/oews_national.xlsx")
    ap.add_argument("--crosswalk", default="data/raw/census_soc_crosswalk.xlsx")
    ap.add_argument("--exposure", default="data/interim/exposure_soc2018.csv")
    ap.add_argument("--out", default="data/interim/occ_measures.csv")
    ap.add_argument("--codes-out", default="data/interim/census_occ_codes.csv",
                    help="every Census code with its SOC major group (for step 05)")
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
    if "task_filled" not in tasks.columns:
        tasks["task_filled"] = 0
    tasks = collapse_onet_soc(tasks, PARTS + ["ln_T", "task_filled"])

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
    df = df.dropna(subset=PARTS)
    if df.empty:
        print("[xwalk] nothing left after merging - check that SOC codes in the three "
              "inputs use the same format (e.g. 15-1252)", file=sys.stderr)
        return 1
    df = fill_employment(df, oews)

    occ = aggregate_to_cps(df)
    no_occ_exp = occ["exposure"].isna().sum()
    if no_occ_exp:
        print(f"[xwalk] {no_occ_exp} CPS occupation codes have task data but no "
              "exposure score (exposure left blank; step 05 drops them)")

    occ = occ.merge(soc_major_by_occ(df), on="cps_occ", how="left")

    # re-close composition after weighted averaging, then recompute ILR
    C = occ[PARTS].to_numpy()
    C = C / C.sum(axis=1, keepdims=True)
    c1, c2, c3, c4 = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
    occ[PARTS] = C
    occ["z1"] = np.sqrt(3 / 4) * np.log(np.cbrt(c1 * c2 * c3) / c4)
    occ["z2"] = np.sqrt(2 / 3) * np.log(np.sqrt(c1 * c3) / c2)
    occ["z3"] = np.sqrt(1 / 2) * np.log(c1 / c3)

    # Exposure terciles (descriptive tables only): employment-weighted cut points
    # WITHIN the knowledge-intensive occupations, so each third holds about a
    # third of the analysed employment. Other codes are placed by the same cuts.
    know = occ["soc_major"].astype(str).str.zfill(2).isin(KNOWLEDGE_MAJOR)
    lo, hi = weighted_terciles(occ.loc[know, "exposure"], occ.loc[know, "emp_total"])
    occ["exposure_tercile"] = np.select(
        [occ["exposure"].isna(), occ["exposure"] <= lo, occ["exposure"] <= hi],
        [None, "low", "mid"], "high")
    print(f"[xwalk] exposure tercile cuts (knowledge occupations, employment-weighted): "
          f"{lo:.3f}, {hi:.3f}")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    occ.to_csv(args.out, index=False)
    codes = census_code_majors(xw_raw)
    codes["has_measures"] = codes["cps_occ"].isin(occ.loc[occ["exposure"].notna(), "cps_occ"])
    codes.to_csv(args.codes_out, index=False)
    print(f"[xwalk] {len(codes)} Census codes ({int(codes['has_measures'].sum())} with "
          f"task and exposure measures) written to {args.codes_out}")
    print(f"[xwalk] {len(occ)} CPS occupation codes written to {args.out}")
    print(occ[["z3", "exposure"]].describe().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
