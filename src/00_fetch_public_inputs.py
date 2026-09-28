"""
00_fetch_public_inputs.py — download the free public inputs used by steps 03 and 05.

    data/raw/oews_national.xlsx          BLS OEWS national estimates (pre-period year)
    data/raw/census_soc_crosswalk.xlsx   Census 2018 occupation code list with SOC crosswalk
    data/raw/cpi_u.csv                   CPI-U all items, NSA (BLS series CUUR0000SA0),
                                         reshaped to columns year,month,cpi
    data/raw/soc_2010_to_2018_crosswalk.xlsx
                                         BLS SOC 2010 -> 2018 crosswalk (used by step
                                         00b for exposure measures coded on SOC 2010)
    data/raw/eloundou_occ_level.csv      Eloundou et al. (2024) occupation-level GPT
                                         exposure (primary measure), from the authors'
                                         repository at a pinned commit; the SHA-256 is
                                         checked so the input cannot change silently

    data/raw/lm_aioe.xlsx                Felten, Raj & Seamans (2023) Language-Modeling
                                         AIOE (robustness measure), from the authors'
                                         repository at a pinned commit, SHA-256 checked

OEWS year: the default is 2021, the first May estimates published entirely on
SOC 2018. May 2019 and 2020 use hybrid codes (e.g. 15-1256 in place of 15-1252
Software Developers and 15-1253 QA Testers), which leaves those detailed
occupations without employment weights in step 03.

BLS rejects requests without a descriptive User-Agent that includes a contact
address, so pass --email (or set BLS_CONTACT_EMAIL).

Every download is recorded (url, sha256, bytes, UTC time) in
data/raw/_public_inputs_manifest.json for the methodology appendix.

Usage:
    python src/00_fetch_public_inputs.py --email you@example.com
    python src/00_fetch_public_inputs.py --oews-year 2019 --email you@example.com
    python src/00_fetch_public_inputs.py --only cpi --email you@example.com
"""
import argparse
import hashlib
import io
import json
import os
import pathlib
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone

OEWS_URL = "https://www.bls.gov/oes/special-requests/oesm{yy}nat.zip"
XWALK_URL = ("https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/"
             "2018-occupation-code-list-and-crosswalk.xlsx")
CPI_URL = "https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems"
CPI_SERIES = "CUUR0000SA0"
SOC_XWALK_URL = "https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx"
ELOUNDOU_COMMIT = "0471612fef3cc22b74fb884d27bff9dbd3770582"
ELOUNDOU_URL = ("https://raw.githubusercontent.com/openai/GPTs-are-GPTs/"
                f"{ELOUNDOU_COMMIT}/data/occ_level.csv")
ELOUNDOU_SHA256 = "40c74f53de40aec91c0017d80690cbba915f83a8bb414bcf2f884692f1749acb"
LMAIOE_COMMIT = "adca5fc2cd0e9a659ff05278b7fa7a53f4f324c1"
LMAIOE_URL = ("https://raw.githubusercontent.com/AIOE-Data/AIOE/"
              f"{LMAIOE_COMMIT}/Language%20Modeling%20AIOE%20and%20AIIE.xlsx")
LMAIOE_SHA256 = "ccdd1fb916dfa404914367eafde7c00b7148ea86f18fe616240bc85cf6131c8b"


def get(url: str, email: str) -> bytes:
    ua = f"PipelineParadox-DBA-research/1.0 (academic use; {email})"
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def record(manifest: dict, name: str, url: str, blob: bytes) -> None:
    manifest[name] = {"url": url, "sha256": hashlib.sha256(blob).hexdigest(),
                      "bytes": len(blob),
                      "downloaded_utc": datetime.now(timezone.utc).isoformat()}


def fetch_oews(raw: pathlib.Path, year: int, email: str, url: str | None, manifest: dict):
    url = url or OEWS_URL.format(yy=f"{year % 100:02d}")
    blob = get(url, email)
    record(manifest, "oews_national", url, blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        xl = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls"))
              and "national" in n.lower() and "field" not in n.lower()]
        if not xl:
            raise ValueError(f"no national workbook in {url}: {zf.namelist()}")
        (raw / "oews_national.xlsx").write_bytes(zf.read(xl[0]))
    print(f"[fetch] OEWS {year} national -> data/raw/oews_national.xlsx ({xl[0]})")


def fetch_crosswalk(raw: pathlib.Path, email: str, url: str | None, manifest: dict):
    url = url or XWALK_URL
    blob = get(url, email)
    record(manifest, "census_soc_crosswalk", url, blob)
    (raw / "census_soc_crosswalk.xlsx").write_bytes(blob)
    print("[fetch] Census 2018 crosswalk -> data/raw/census_soc_crosswalk.xlsx")


def fetch_soc_crosswalk(raw: pathlib.Path, email: str, url: str | None, manifest: dict):
    url = url or SOC_XWALK_URL
    blob = get(url, email)
    record(manifest, "soc_2010_to_2018_crosswalk", url, blob)
    (raw / "soc_2010_to_2018_crosswalk.xlsx").write_bytes(blob)
    print("[fetch] BLS SOC 2010->2018 crosswalk -> data/raw/soc_2010_to_2018_crosswalk.xlsx")


def fetch_eloundou(raw: pathlib.Path, email: str, url: str | None, manifest: dict):
    url = url or ELOUNDOU_URL
    blob = get(url, email)
    sha = hashlib.sha256(blob).hexdigest()
    if url == ELOUNDOU_URL and sha != ELOUNDOU_SHA256:
        raise ValueError(f"checksum mismatch for the pinned Eloundou file: {sha}")
    record(manifest, "eloundou_occ_level", url, blob)
    (raw / "eloundou_occ_level.csv").write_bytes(blob)
    print(f"[fetch] Eloundou et al. occ_level.csv @ {ELOUNDOU_COMMIT[:7]} "
          "-> data/raw/eloundou_occ_level.csv (checksum verified)")


def fetch_lmaioe(raw: pathlib.Path, email: str, url: str | None, manifest: dict):
    url = url or LMAIOE_URL
    blob = get(url, email)
    sha = hashlib.sha256(blob).hexdigest()
    if url == LMAIOE_URL and sha != LMAIOE_SHA256:
        raise ValueError(f"checksum mismatch for the pinned LM-AIOE file: {sha}")
    record(manifest, "lm_aioe", url, blob)
    (raw / "lm_aioe.xlsx").write_bytes(blob)
    print(f"[fetch] Felten et al. LM-AIOE @ {LMAIOE_COMMIT[:7]} "
          "-> data/raw/lm_aioe.xlsx (checksum verified)")


def parse_cpi(blob: bytes):
    """BLS cu.data.1.AllItems -> monthly CPI-U (year, month, cpi) from 2015.

    Months BLS did not publish appear as "-" in the file. They are kept with a
    missing cpi rather than filled in: how to treat them is an analysis
    decision.
    """
    import pandas as pd
    df = pd.read_csv(io.BytesIO(blob), sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["series_id"].str.strip() == CPI_SERIES]
    df = df[df["period"].str.match(r"^M(0[1-9]|1[0-2])$")]          # drop M13 annual avg
    out = pd.DataFrame({"year": df["year"].astype(int),
                        "month": df["period"].str[1:].astype(int),
                        "cpi": pd.to_numeric(df["value"].str.strip(), errors="coerce")})
    return out[out["year"] >= 2015].sort_values(["year", "month"])


def fetch_cpi(raw: pathlib.Path, email: str, url: str | None, manifest: dict):
    url = url or CPI_URL
    blob = get(url, email)
    record(manifest, "cpi_u", url, blob)
    out = parse_cpi(blob)
    out.to_csv(raw / "cpi_u.csv", index=False)
    print(f"[fetch] CPI-U {CPI_SERIES} {out['year'].min()}-{out['year'].max()} "
          f"-> data/raw/cpi_u.csv ({len(out)} months)")
    missing = out[out["cpi"].isna()]
    for y, m in zip(missing["year"], missing["month"]):
        print(f"[fetch] WARNING: CPI-U not published for {y}-{m:02d}; left blank in cpi_u.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--oews-year", type=int, default=2021,
                    help="pre-period OEWS year used for employment weights "
                         "(2021 = first year fully on SOC 2018)")
    ap.add_argument("--email", default=os.environ.get("BLS_CONTACT_EMAIL"))
    ap.add_argument("--only", choices=["oews", "crosswalk", "cpi", "soc", "eloundou", "lmaioe"],
                    nargs="+")
    ap.add_argument("--oews-url", help="override if BLS moves the file")
    ap.add_argument("--crosswalk-url", help="override if Census moves the file")
    ap.add_argument("--cpi-url", help="override if BLS moves the file")
    ap.add_argument("--soc-url", help="override if BLS moves the SOC crosswalk")
    ap.add_argument("--eloundou-url", help="override the pinned Eloundou file (no "
                    "checksum check)")
    ap.add_argument("--lmaioe-url", help="override the pinned LM-AIOE file (no "
                    "checksum check)")
    args = ap.parse_args()

    # Only the BLS and Census downloads need a contact address; the Eloundou
    # and LM-AIOE files come from GitHub and can be fetched without one.
    if not args.email and set(args.only or ["oews"]) - {"eloundou", "lmaioe"}:
        print("[fetch] pass --email or set BLS_CONTACT_EMAIL (BLS requires a contact "
              "address in the User-Agent)", file=sys.stderr)
        return 1

    raw = pathlib.Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    mpath = raw / "_public_inputs_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}

    jobs = {"oews": lambda: fetch_oews(raw, args.oews_year, args.email, args.oews_url, manifest),
            "crosswalk": lambda: fetch_crosswalk(raw, args.email, args.crosswalk_url, manifest),
            "cpi": lambda: fetch_cpi(raw, args.email, args.cpi_url, manifest),
            "soc": lambda: fetch_soc_crosswalk(raw, args.email, args.soc_url, manifest),
            "eloundou": lambda: fetch_eloundou(raw, args.email, args.eloundou_url,
                                               manifest),
            "lmaioe": lambda: fetch_lmaioe(raw, args.email, args.lmaioe_url, manifest)}
    failed = []
    for name in args.only or jobs:
        try:
            jobs[name]()
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"[fetch] {name} FAILED: {type(exc).__name__}: {exc}\n"
                  f"        download it manually (see data/README.md) or re-run with "
                  f"--{name}-url <link>", file=sys.stderr)

    mpath.write_text(json.dumps(manifest, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
