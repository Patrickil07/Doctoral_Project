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
    args = ap.parse_args()

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
    original_argv = sys.argv
    sys.argv = ['colab_kernel_launcher.py', '--start', '2019-01', '--end', '2025-12', '--outdir', 'data/raw/ipums']
    try:
        ipums_dir = pathlib.Path("data/raw/ipums")
        existing_xml = list(ipums_dir.glob("*.xml")) if ipums_dir.exists() else []
        if existing_xml:
            print(f"[ipums] Extract codebook already present in {ipums_dir}: {existing_xml[0].name}")
            print("[ipums] Skipping download. (Delete file if you wish to re-request extract).")
        elif not os.environ.get("IPUMS_API_KEY"):
            print("[ipums] ⚠️ IPUMS_API_KEY environment variable not set.")
            print("        To request new extract: os.environ['IPUMS_API_KEY'] = 'YOUR_KEY'")
            print("        Or place existing extract (.xml + .dat.gz) directly in data/raw/ipums/.")
        else:
            exit_code = main()
            if exit_code != 0:
                print(f"[ipums] Finished with exit code {exit_code}")
    finally:
        sys.argv = original_argv
