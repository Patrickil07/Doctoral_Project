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
    python src/04_ipums_extract.py --start 2020-01 --end 2025-12
    python src/04_ipums_extract.py --force        # re-request even if present
"""
import argparse
import os
import pathlib
import sys

VARS = [
    # Only variables that 05_build_sample.py or 07_estimate.py use. IPUMS adds
    # its own preselected identifiers and weights (SERIAL, PERNUM, WTFINL, ...).
    "YEAR", "MONTH", "EARNWT",                    # time; ORG earnings weight
    "STATEFIP", "AGE", "SEX", "EDUC",             # controls, age bands, state FE
    "EMPSTAT", "CLASSWKR", "IND", "OCC", "OCCSOC",  # sample rules, industry FE, merge key
    "EARNWEEK", "EARNWEEK2", "QEARNWEEK",         # earnings; allocation flag
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


def available_samples(client, collection: str = "cps", page_size: int = 500) -> set[str]:
    """Every sample id IPUMS lists for `collection`, following all result pages.

    ipumspy's get_all_sample_info reads only the first page, which returns an
    arbitrary subset of CPS samples; relying on it drops real months.
    """
    names: set[str] = set()
    page = 1
    while True:
        r = client.get(f"{client.base_url}/metadata/samples",
                       params={"collection": collection, "version": client.api_version,
                               "pageNumber": page, "pageSize": page_size}).json()
        data = r.get("data") or []
        names.update(item["name"] for item in data)
        total = r.get("totalCount")
        # the server may cap pageSize, so prefer totalCount to decide when to stop
        done = len(names) >= total if total is not None else len(data) < page_size
        if not data or done:
            return names
        page += 1


def split_available(wanted: list[str], available) -> tuple[list[str], list[str]]:
    """Split requested sample ids into those IPUMS offers and those it does not."""
    have = set(available)
    return [s for s in wanted if s in have], [s for s in wanted if s not in have]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01",
                    help="2020-01 = first month on 2018 Census occupation codes")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--outdir", default="data/raw/ipums")
    ap.add_argument("--description", default="Pipeline Paradox CPS ORG 2020-2025")
    ap.add_argument("--max-missing", type=int, default=2,
                    help="stop if more requested months than this are unavailable")
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

    client = IpumsApiClient(key)
    wanted = month_samples(args.start, args.end)
    samples, missing = split_available(wanted, available_samples(client))
    for s in missing:
        print(f"[ipums] WARNING: IPUMS has no sample {s}; that month is not requested")
    if len(missing) > args.max_missing:
        print(f"[ipums] {len(missing)} of {len(wanted)} months are not listed by IPUMS "
              f"(limit {args.max_missing}); stopping so months are not dropped silently. "
              "Check the list above against the IPUMS CPS sample page, or raise "
              "--max-missing if the gaps are real.", file=sys.stderr)
        return 1
    print(f"[ipums] requesting {len(samples)} monthly samples "
          f"({samples[0]} to {samples[-1]}), {len(VARS)} variables")

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
