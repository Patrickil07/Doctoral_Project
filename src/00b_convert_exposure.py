"""
00b_convert_exposure.py — build the occupation exposure measure on SOC 2018 codes.

Steps 03 onwards key everything on 6-digit SOC 2018. This step produces that
file from the chosen exposure source.

Primary measure (default): Eloundou, Manning, Mishkin & Rock (2024), "GPTs are
GPTs", occupation-level file occ_level.csv (fetched and checksum-verified by
step 00). It is on O*NET-SOC 2019 codes (SOC 2018 based, e.g. 15-1211.01).
Columns: human_rating_* (annotators) and dv_rating_* (GPT-4), each as
alpha = E1, beta = E1 + 0.5*E2, gamma = E1 + E2. The default is
human_rating_beta; --column switches rater or definition for robustness.

O*NET-SOC detail codes are collapsed to 6-digit SOC with the same rule step 03
applies to the task measures: the score of the main occupation (.00) where it
exists, otherwise the unweighted mean of the detail codes.

Measures on SOC 2010 codes (Felten, Raj & Seamans' AIOE and LM-AIOE) need
--source-soc 2010. They are mapped with the BLS 2010-to-2018 SOC crosswalk:
  * one 2010 code -> several 2018 codes (a split): each inherits the score;
  * several 2010 codes -> one 2018 code (a merge): unweighted mean.
The log lists every 2018 code built from more than one source code.

Scale: Eloundou scores are shares in [0, 1]; AIOE scores are standardised.
Coefficients on exposure are therefore not comparable across measures
without rescaling.

Output
  data/interim/exposure_soc2018.csv    columns: soc2018,exposure,n_source

Usage:
    python src/00b_convert_exposure.py                               # primary
    python src/00b_convert_exposure.py --column dv_rating_beta \
        --out data/interim/exposure_gpt4beta_soc2018.csv              # GPT-4 rater
    python src/00b_convert_exposure.py --exposure data/raw/lm_aioe.xlsx \
        --column "Language Modeling AIOE" --source-soc 2010 \
        --out data/interim/exposure_lmaioe_soc2018.csv                # LM-AIOE
"""
import argparse
import pathlib
import sys

import pandas as pd

SOC = r"^\d{2}-\d{4}$"
ONET_SOC = r"^\d{2}-\d{4}\.\d{2}$"


def collapse_onet_soc(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Keep only the main O*NET-SOC occupation (.00) of each 6-digit SOC that has
    one, and cut every O*NET-SOC code to 6 digits. SOCs without a .00 code keep
    all their detail codes (averaged later). Rows already on 6-digit SOC pass
    through unchanged.

    Averaging niche detail occupations (e.g. Chief Sustainability Officers)
    with equal weight to the main occupation (Chief Executives) would give them
    far more weight than their employment warrants.
    """
    code = df[col].astype(str)
    detail = code.str.match(ONET_SOC, na=False)
    six = code.where(~detail, code.str.slice(0, 7))
    main = detail & code.str.endswith(".00")
    has_main = set(six[main])
    keep = ~detail | main | ~six.isin(has_main)
    out = df[keep].copy()
    out[col] = six[keep]
    return out


def load_exposure(path: pathlib.Path, column: str, sheet: str | None = None
                  ) -> pd.DataFrame:
    """Read an exposure file (.csv or .xlsx) into columns soc, exposure, n_source.

    The code column is the first whose name contains 'soc'. O*NET-SOC detail
    codes (15-1211.01) are collapsed to 6-digit SOC (15-1211) by `collapse_onet_soc`.
    """
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    code = next((c for c in df.columns if "soc" in c.lower()), None)
    if code is None or column not in df.columns:
        raise ValueError(f"{path}: need a SOC code column and the score column "
                         f"'{column}' (see --column); found {list(df.columns)}")
    print(f"[exposure] {path.name}: codes from '{code}', scores from '{column}'")
    out = pd.DataFrame({"soc": df[code].astype(str).str.strip(),
                        "exposure": pd.to_numeric(df[column], errors="coerce")})

    detail = out["soc"].str.match(ONET_SOC, na=False)
    if detail.any():
        n_before = int(detail.sum())
        out = collapse_onet_soc(out, "soc")
        print(f"[exposure] {n_before} O*NET-SOC codes collapsed to 6-digit SOC "
              f"(.00 score where it exists, else the mean of the detail codes)")

    bad = ~out["soc"].str.match(SOC, na=False) | out["exposure"].isna()
    if bad.any():
        print(f"[exposure] dropping {int(bad.sum())} rows without a SOC code "
              f"or numeric exposure")
    return (out[~bad].groupby("soc")
                     .agg(exposure=("exposure", "mean"), n_source=("exposure", "size"))
                     .reset_index())


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
    """Map a SOC 2010 exposure table to SOC 2018 (split = copy, merge = mean)."""
    m = xw.merge(exp.rename(columns={"soc": "soc2010"}), on="soc2010", how="inner")
    out = (m.groupby("soc2018")
             .agg(exposure=("exposure", "mean"), n_source=("soc2010", "nunique"))
             .reset_index())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure", default="data/raw/eloundou_occ_level.csv")
    ap.add_argument("--column", default="human_rating_beta",
                    help="score column (default: Eloundou human-rated beta)")
    ap.add_argument("--sheet", help="Excel sheet (default: first)")
    ap.add_argument("--source-soc", choices=["2018", "2010"], default="2018",
                    help="SOC vintage of the source codes; 2010 triggers conversion")
    ap.add_argument("--crosswalk", default="data/raw/soc_2010_to_2018_crosswalk.xlsx")
    ap.add_argument("--out", default="data/interim/exposure_soc2018.csv")
    args = ap.parse_args()

    src = pathlib.Path(args.exposure)
    if not src.exists():
        print(f"[exposure] missing {src}; run step 00 (it downloads the Eloundou "
              "et al. file) or place your source there", file=sys.stderr)
        return 1
    exp = load_exposure(src, args.column, args.sheet)
    print(f"[exposure] {len(exp)} 6-digit SOC {args.source_soc} occupations read")

    if args.source_soc == "2018":
        out = exp.rename(columns={"soc": "soc2018"})
    else:
        xw_path = pathlib.Path(args.crosswalk)
        if not xw_path.exists():
            print(f"[exposure] missing {xw_path}; run step 00 (it downloads the BLS "
                  "SOC 2010-to-2018 crosswalk)", file=sys.stderr)
            return 1
        xw = load_soc_crosswalk(xw_path)
        out = convert(exp[["soc", "exposure"]], xw)
        unmatched = sorted(set(exp["soc"]) - set(xw["soc2010"]))
        if unmatched:
            print(f"[exposure] {len(unmatched)} SOC 2010 codes are not in the crosswalk "
                  f"and are dropped: {unmatched[:10]}")
        merged = out[out["n_source"] > 1]
        print(f"[exposure] {len(out)} SOC 2018 codes; {len(merged)} are the mean of "
              f"several 2010 codes (listed below — check and report)")
        for r in merged.itertuples(index=False):
            codes = sorted(xw.loc[xw["soc2018"] == r.soc2018, "soc2010"])
            print(f"        {r.soc2018} <- {', '.join(codes)}")

    print(f"[exposure] {args.column}: mean {out['exposure'].mean():.3f}, "
          f"sd {out['exposure'].std():.3f}, range [{out['exposure'].min():.3f}, "
          f"{out['exposure'].max():.3f}]")
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[exposure] written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
