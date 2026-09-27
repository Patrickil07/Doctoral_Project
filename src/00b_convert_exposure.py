"""
00b_convert_exposure.py — put an occupation exposure measure on SOC 2018 codes.

Steps 03 onwards key everything on SOC 2018. Several AI-exposure measures are
published on SOC 2010 codes, including Felten, Raj & Seamans' AIOE and
Language-Modeling AIOE. Merging those directly on SOC 2018 silently drops every
occupation whose code changed in 2018 — most of the computer occupations
(15-11xx -> 15-12xx) among them.

This script maps a SOC 2010 exposure file to SOC 2018 with the BLS
2010-to-2018 SOC crosswalk:
  * one 2010 code -> several 2018 codes (a split): each 2018 code inherits
    the 2010 score;
  * several 2010 codes -> one 2018 code (a merge): the 2018 code gets the
    unweighted mean of the 2010 scores.
Both rules are choices to report in the methodology; the log lists every
2018 code built from more than one 2010 code so they can be checked.

A measure already on SOC 2018 (e.g. Eloundou et al. 2024, O*NET-SOC 2019)
needs no conversion: pass --already-2018 to validate and copy it.

Inputs
  data/raw/lm_aioe.xlsx                       Felten, Raj & Seamans LM-AIOE (SOC 2010),
                                              or any .csv/.xlsx with a SOC column and
                                              a score column (--column)
  data/raw/soc_2010_to_2018_crosswalk.xlsx    BLS (fetched by step 00)
Output
  data/interim/exposure_soc2018.csv           columns: soc2018,exposure,n_soc2010

Usage:
    python src/00b_convert_exposure.py
    python src/00b_convert_exposure.py --exposure data/raw/AIOE_DataAppendix.xlsx \
        --sheet "Appendix A" --column AIOE --out data/interim/exposure_aioe_soc2018.csv
    python src/00b_convert_exposure.py --exposure data/raw/eloundou_soc2018.csv --already-2018
"""
import argparse
import pathlib
import sys

import pandas as pd

SOC = r"^\d{2}-\d{4}$"


def load_exposure(path: pathlib.Path, sheet: str | None = None,
                  column: str | None = None) -> pd.DataFrame:
    """Read an exposure file (.csv or .xlsx) into columns soc, exposure.

    The SOC column is the first whose name contains 'soc'. The score column is
    --column if given, else 'exposure', else the last numeric column.
    """
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    code = next((c for c in df.columns if "soc" in c.lower()), None)
    if column:
        value = column
    elif "exposure" in [c.lower() for c in df.columns]:
        value = next(c for c in df.columns if c.lower() == "exposure")
    else:
        numeric = [c for c in df.columns
                   if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
        value = numeric[-1] if numeric else None
    if code is None or value not in df.columns:
        raise ValueError(f"{path}: need a SOC code column and a score column "
                         f"(pass --column); found {list(df.columns)}")
    print(f"[exposure] {path.name}: codes from '{code}', scores from '{value}'")
    out = pd.DataFrame({"soc": df[code].astype(str).str.strip(),
                        "exposure": pd.to_numeric(df[value], errors="coerce")})
    bad = ~out["soc"].str.match(SOC, na=False) | out["exposure"].isna()
    if bad.any():
        print(f"[exposure] dropping {int(bad.sum())} rows without a detailed SOC code "
              f"or numeric exposure")
    return out[~bad].drop_duplicates("soc")


def load_soc_crosswalk(path: pathlib.Path) -> pd.DataFrame:
    """Read a 2010->2018 SOC crosswalk whose header sits below title rows.

    Finds the header row containing both a '2010 SOC code' and a '2018 SOC code'
    column, so it works for the BLS file regardless of how many title rows it has.
    """
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        raw = xl.parse(sheet, header=None, dtype=str)
        for i in range(min(20, len(raw))):
            cells = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
            c10 = next((j for j, v in enumerate(cells) if "2010" in v and "soc" in v
                        and "code" in v), None)
            c18 = next((j for j, v in enumerate(cells) if "2018" in v and "soc" in v
                        and "code" in v), None)
            if c10 is None or c18 is None:
                continue
            body = raw.iloc[i + 1:, [c10, c18]]
            body.columns = ["soc2010", "soc2018"]
            body = body.apply(lambda s: s.str.strip())
            body["soc2010"] = body["soc2010"].ffill()        # continuation rows
            body = body[body["soc2010"].str.match(SOC, na=False)
                        & body["soc2018"].str.match(SOC, na=False)]
            if len(body) >= 5:
                print(f"[exposure] crosswalk sheet '{sheet}', header row {i}: "
                      f"{len(body)} SOC 2010 -> 2018 pairs")
                return body.drop_duplicates().reset_index(drop=True)
    raise ValueError(f"no '2010 SOC code' / '2018 SOC code' header found in {path}")


def convert(exp: pd.DataFrame, xw: pd.DataFrame) -> pd.DataFrame:
    m = xw.merge(exp.rename(columns={"soc": "soc2010"}), on="soc2010", how="inner")
    out = (m.groupby("soc2018")
             .agg(exposure=("exposure", "mean"), n_soc2010=("soc2010", "nunique"))
             .reset_index())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure", default="data/raw/lm_aioe.xlsx")
    ap.add_argument("--sheet", help="Excel sheet (default: first)")
    ap.add_argument("--column", help="score column (default: 'exposure' or last numeric)")
    ap.add_argument("--crosswalk", default="data/raw/soc_2010_to_2018_crosswalk.xlsx")
    ap.add_argument("--out", default="data/interim/exposure_soc2018.csv")
    ap.add_argument("--already-2018", action="store_true",
                    help="input is already on SOC 2018: validate and copy")
    args = ap.parse_args()

    exp = load_exposure(pathlib.Path(args.exposure), args.sheet, args.column)
    print(f"[exposure] {len(exp)} occupations read from {args.exposure}")

    if args.already_2018:
        out = exp.rename(columns={"soc": "soc2018"}).assign(n_soc2010=pd.NA)
    else:
        xw_path = pathlib.Path(args.crosswalk)
        if not xw_path.exists():
            print(f"[exposure] missing {xw_path}; run step 00 (it downloads the BLS "
                  "SOC 2010-to-2018 crosswalk)", file=sys.stderr)
            return 1
        xw = load_soc_crosswalk(xw_path)
        out = convert(exp, xw)
        unmatched = sorted(set(exp["soc"]) - set(xw["soc2010"]))
        if unmatched:
            print(f"[exposure] {len(unmatched)} SOC 2010 codes are not in the crosswalk "
                  f"and are dropped: {unmatched[:10]}")
        merged = out[out["n_soc2010"] > 1]
        print(f"[exposure] {len(out)} SOC 2018 codes; {len(merged)} are the mean of "
              f"several 2010 codes (listed below — check and report)")
        for r in merged.itertuples(index=False):
            src = sorted(xw.loc[xw["soc2018"] == r.soc2018, "soc2010"])
            print(f"        {r.soc2018} <- {', '.join(src)}")

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[exposure] written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
