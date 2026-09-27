"""
00_fetch_public_inputs.py — download the free public inputs used by steps 03 and 05.

    data/raw/oews_national.xlsx          BLS OEWS national estimates (pre-period year)
    data/raw/census_soc_crosswalk.xlsx   Census 2018 occupation code list with SOC crosswalk
    data/raw/cpi_u.csv                   CPI-U all items, NSA (BLS series CUUR0000SA0),
                                         reshaped to columns year,month,cpi

NOT fetched here: data/raw/exposure_soc.csv (columns soc2018,exposure). The GenAI
exposure measure is a methodological choice; build it from your chosen source
and record that source in data/README.md.

BLS rejects requests without a descriptive User-Agent that includes a contact
address, so pass --email (or set BLS_CONTACT_EMAIL).

Every download is recorded (url, sha256, bytes, UTC time) in
data/raw/_public_inputs_manifest.json for the methodology appendix.

Usage:
    python src/00_fetch_public_inputs.py --email you@example.com
    python src/00_fetch_public_inputs.py --oews-year 2021 --email you@example.com
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


def fetch_cpi(raw: pathlib.Path, email: str, url: str | None, manifest: dict):
    import pandas as pd
    url = url or CPI_URL
    blob = get(url, email)
    record(manifest, "cpi_u", url, blob)
    df = pd.read_csv(io.BytesIO(blob), sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["series_id"].str.strip() == CPI_SERIES]
    df = df[df["period"].str.match(r"^M(0[1-9]|1[0-2])$")]          # drop M13 annual avg
    out = pd.DataFrame({"year": df["year"].astype(int),
                        "month": df["period"].str[1:].astype(int),
                        "cpi": pd.to_numeric(df["value"].str.strip())})
    out = out[out["year"] >= 2015].sort_values(["year", "month"])
    out.to_csv(raw / "cpi_u.csv", index=False)
    print(f"[fetch] CPI-U {CPI_SERIES} {out['year'].min()}-{out['year'].max()} "
          f"-> data/raw/cpi_u.csv ({len(out)} months)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--oews-year", type=int, default=2019,
                    help="pre-period OEWS year used for employment weights")
    ap.add_argument("--email", default=os.environ.get("BLS_CONTACT_EMAIL"))
    ap.add_argument("--only", choices=["oews", "crosswalk", "cpi"], nargs="+")
    ap.add_argument("--oews-url", help="override if BLS moves the file")
    ap.add_argument("--crosswalk-url", help="override if Census moves the file")
    ap.add_argument("--cpi-url", help="override if BLS moves the file")
    args = ap.parse_args()

    if not args.email:
        print("[fetch] pass --email or set BLS_CONTACT_EMAIL (BLS requires a contact "
              "address in the User-Agent)", file=sys.stderr)
        return 1

    raw = pathlib.Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    mpath = raw / "_public_inputs_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}

    jobs = {"oews": lambda: fetch_oews(raw, args.oews_year, args.email, args.oews_url, manifest),
            "crosswalk": lambda: fetch_crosswalk(raw, args.email, args.crosswalk_url, manifest),
            "cpi": lambda: fetch_cpi(raw, args.email, args.cpi_url, manifest)}
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
    if not (raw / "exposure_soc.csv").exists():
        print("[fetch] reminder: data/raw/exposure_soc.csv must be built by hand "
              "(see data/README.md)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
