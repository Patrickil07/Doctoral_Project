"""
04_ipums_extract.py — request and download a CPS Outgoing Rotation Group extract
through the official IPUMS API.

This uses the IPUMS API, not scraping. IPUMS terms of use prohibit automated
harvesting of the site, and the API is the sanctioned route. You need:
  1. A free IPUMS account:      https://uma.pop.umn.edu/usa/user/new
  2. An API key:                https://account.ipums.org/api_keys
  3. export IPUMS_API_KEY=...   (never commit the key)

The extract may take several minutes to build server-side; this script polls
until it is ready, then downloads the data and DDI codebook.

Usage:
    export IPUMS_API_KEY=xxxxxxxx
    python src/04_ipums_extract.py --start 2019-01 --end 2025-12
    python src/04_ipums_extract.py --force        # re-request even if present
"""
import argparse
import os
import pathlib
import sys

VARS = [
    # identifiers / weights
    "YEAR", "MONTH", "CPSID", "CPSIDP", "ASECFLAG", "MISH", "EARNWT", "WTFINL",
    # geography & demographics
    "STATEFIP", "METFIPS", "AGE", "SEX", "RACE", "HISPAN", "EDUC",
    # employment
    "EMPSTAT", "LABFORCE", "CLASSWKR", "IND", "OCC", "OCC2010", "OCCSOC",
    "UHRSWORKORG", "PAIDHOUR", "UNION",
    # earnings (ORG) + allocation flags for the Hirsch-Schumacher exclusion
    "EARNWEEK", "EARNWEEK2", "HOURWAGE", "HOURWAGE2",
    "QEARNWEEK", "QHOURWAGE", "OTPAY",
]


def month_samples(start: str, end: str) -> list[str]:
    ys, ms = (int(x) for x in start.split("-"))
    ye, me = (int(x) for x in end.split("-"))
    out = []
    y, m = ys, ms
    while (y, m) <= (ye, me):
        out.append(f"cps{y}_{m:02d}b")  # basic monthly sample ids
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--outdir", default="data/raw/ipums")
    ap.add_argument("--description", default="Pipeline Paradox CPS ORG 2019-2025")
    ap.add_argument("--force", action="store_true",
                    help="submit a new extract even if one is already downloaded")
    args = ap.parse_args()

    existing = sorted(pathlib.Path(args.outdir).glob("*.xml"))
    if existing and not args.force:
        print(f"[ipums] extract codebook already present: {existing[0]}\n"
              "        skipping; pass --force to request a new extract")
        return 0

    key = os.environ.get("IPUMS_API_KEY")
    if not key:
        print("[ipums] set IPUMS_API_KEY first (https://account.ipums.org/api_keys)",
              file=sys.stderr)
        return 1

    try:
        from ipumspy import IpumsApiClient, MicrodataExtract
    except ImportError:
        print("[ipums] pip install ipumspy", file=sys.stderr)
        return 1

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    samples = month_samples(args.start, args.end)
    print(f"[ipums] requesting {len(samples)} monthly samples, {len(VARS)} variables")
    print("[ipums] NOTE: October 2025 was not collected (federal shutdown) and will "
          "be absent; this is expected — see proposal Section 6.6.")

    client = IpumsApiClient(key)
    extract = MicrodataExtract(
        collection="cps",
        description=args.description,
        samples=samples,
        variables=VARS,
    )

    client.submit_extract(extract)
    print(f"[ipums] submitted extract #{extract.extract_id}; waiting…")
    client.wait_for_extract(extract)
    client.download_extract(extract, download_dir=outdir)

    print(f"[ipums] downloaded to {outdir}")
    print(f"[ipums] RECORD THIS IN THE METHODOLOGY APPENDIX: "
          f"IPUMS CPS extract #{extract.extract_id}, collection=cps")
    print("[ipums] cite the exact IPUMS CPS version shown in the DDI codebook")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
