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
import re
import sys

VARS = [
    # Only variables that 05_build_sample.py or 07_estimate.py use. IPUMS adds
    # its own preselected identifiers and weights (SERIAL, PERNUM, WTFINL, ...).
    # The SOC major group comes from the Census crosswalk (step 03): IPUMS CPS
    # has no OCCSOC variable.
    "YEAR", "MONTH", "EARNWT",                    # time; ORG earnings weight
    "STATEFIP", "AGE", "SEX", "EDUC",             # controls, age bands, state FE
    "EMPSTAT", "CLASSWKR", "IND", "OCC",          # sample rules, industry FE, merge key
    "EARNWEEK", "EARNWEEK2",                      # weekly earnings
    "UHRSWORKORG",                                # usual weekly hours (ORG): full-time rule
    "WTFINL",                                     # final person weight (RQ2, all rotation groups)
]
# Requested if IPUMS accepts the name; dropped with a warning if it does not.
OPTIONAL_VARS: list[str] = []
# IPUMS does not accept allocation flags as variable names (QEARNWEE was
# rejected as "Invalid mnemonic"): they come with the data-quality-flags option
# of each variable. Step 05 drops records whose earnings were allocated.
FLAGGED_VARS = ["EARNWEEK", "EARNWEEK2", "UHRSWORKORG"]


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


def available_samples(client, collection: str = "cps", page_size: int = 500) -> dict[str, str]:
    """Every sample IPUMS lists for `collection` as {id: description}, all pages.

    ipumspy's get_all_sample_info reads only the first page of results.
    """
    info: dict[str, str] = {}
    page = 1
    while True:
        r = client.get(f"{client.base_url}/metadata/samples",
                       params={"collection": collection, "version": client.api_version,
                               "pageNumber": page, "pageSize": page_size}).json()
        data = r.get("data") or []
        info.update({item["name"]: item.get("description", "") for item in data})
        total = r.get("totalCount")
        # the server may cap pageSize, so prefer totalCount to decide when to stop
        done = len(info) >= total if total is not None else len(data) < page_size
        if not data or done:
            return info
        page += 1


MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def pick_monthly_samples(wanted: list[str], info: dict[str, str]) -> tuple[list[str], list[str]]:
    """Map each requested month (cpsYYYY_MMb) to the sample IPUMS actually offers.

    IPUMS names a monthly sample cpsYYYY_MMb, or cpsYYYY_MMs when that month also
    carried a supplement; both hold the full basic monthly survey and are
    described as "IPUMS-CPS, <Month> <Year>". The March ASEC ("IPUMS-CPS, ASEC
    <Year>") is a different survey and is never picked. Returns (sample ids,
    months with no monthly sample).
    """
    by_desc: dict[str, str] = {}
    for name, desc in sorted(info.items(), key=lambda kv: not kv[0].endswith("b")):
        by_desc.setdefault(desc.strip(), name)          # prefer the ...b sample
    picked, missing = [], []
    for s in wanted:
        y, m = int(s[3:7]), int(s[8:10])
        name = by_desc.get(f"IPUMS-CPS, {MONTH_NAMES[m - 1]} {y}")
        (picked if name else missing).append(name or s)
    return picked, missing


def submit_dropping_optional(client, make, required: list[str], optional: list[str]):
    """Submit an extract; if IPUMS rejects only optional variable names, drop them and retry.

    A rejected request creates no extract. Any rejected required variable is an error.
    """
    from ipumspy.api.exceptions import BadIpumsApiRequest
    variables = list(required) + list(optional)
    while True:
        extract = make(variables)
        try:
            client.submit_extract(extract)
            return extract, variables
        except BadIpumsApiRequest as exc:
            # IPUMS words this "Invalid variable name: X" or "Invalid mnemonic: X"
            bad = re.findall(r"Invalid (?:variable name|mnemonic): (\w+)", str(exc))
            if not bad or any(v not in optional for v in bad):
                raise
            for v in bad:
                print(f"[ipums] WARNING: IPUMS does not recognise {v}; requesting without it")
            variables = [v for v in variables if v not in bad]


def fetch_existing(client, extract_id: int, outdir: pathlib.Path) -> int:
    """Download extract `extract_id`; if IPUMS has expired its files, re-submit the
    identical definition (same samples and variables) and download that."""
    import json
    from datetime import datetime, timezone
    used, note = extract_id, "downloaded as is"
    if client.extract_is_expired(extract_id, "cps"):
        extract = client.get_extract_by_id(extract_id, "cps")
        client.submit_extract(extract)
        used, note = extract.extract_id, f"re-submitted from expired #{extract_id}"
        print(f"[ipums] extract #{extract_id} has expired; re-submitted as #{used}, waiting…")
    client.wait_for_extract(used, "cps")
    client.download_extract(used, collection="cps", download_dir=outdir)
    (outdir / "_extract_used.json").write_text(json.dumps({
        "extract_id": used, "requested_id": extract_id, "note": note,
        "downloaded_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"[ipums] IPUMS CPS extract #{used} ({note}) -> {outdir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01",
                    help="2020-01 = first month on 2018 Census occupation codes")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--outdir", default="data/raw/ipums")
    ap.add_argument("--description", default="Pipeline Paradox CPS ORG 2020-2025")
    ap.add_argument("--max-missing", type=int, default=2,
                    help="stop if more requested months than this are unavailable")
    ap.add_argument("--extract-id", type=int,
                    help="download this existing extract (re-submitted if expired) "
                         "instead of building a new request")
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

    if args.extract_id:
        return fetch_existing(IpumsApiClient(key), args.extract_id, outdir)

    client = IpumsApiClient(key)
    wanted = month_samples(args.start, args.end)
    samples, missing = pick_monthly_samples(wanted, available_samples(client))
    for s in missing:
        print(f"[ipums] WARNING: IPUMS has no monthly sample for {s[3:7]}-{s[8:10]}; "
              "that month is not requested")
    if len(missing) > args.max_missing:
        print(f"[ipums] {len(missing)} of {len(wanted)} months are not listed by IPUMS "
              f"(limit {args.max_missing}); stopping so months are not dropped silently. "
              "Check the list above against the IPUMS CPS sample page, or raise "
              "--max-missing if the gaps are real.", file=sys.stderr)
        return 1
    print(f"[ipums] requesting {len(samples)} monthly samples "
          f"({samples[0]} to {samples[-1]}), {len(VARS)} variables")

    def make(variables):
        extract = MicrodataExtract(collection="cps", description=args.description,
                                   samples=samples, variables=variables)
        extract.add_data_quality_flags([v for v in FLAGGED_VARS if v in variables])
        return extract

    extract, used = submit_dropping_optional(client, make, VARS, OPTIONAL_VARS)
    print(f"[ipums] submitted extract #{extract.extract_id} with variables {used}; waiting…")
    client.wait_for_extract(extract)
    client.download_extract(extract, download_dir=outdir)
    import json
    from datetime import datetime, timezone
    (outdir / "_extract_used.json").write_text(json.dumps({
        "extract_id": extract.extract_id, "requested_id": None,
        "note": f"new request; variables {used}; data quality flags for {FLAGGED_VARS}",
        "downloaded_utc": datetime.now(timezone.utc).isoformat()}, indent=2))

    print(f"[ipums] downloaded to {outdir}")
    print(f"[ipums] RECORD THIS IN THE METHODOLOGY APPENDIX: "
          f"IPUMS CPS extract #{extract.extract_id}, collection=cps")
    print("[ipums] cite the exact IPUMS CPS version shown in the DDI codebook")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
