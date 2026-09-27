"""
06_usajobs_historic.py — retrieve historical federal job announcements.

Public bulk endpoints, NO authentication required:
    GET /api/historicjoa                  structured fields (grade, salary, dates, series)
    GET /api/historicjoa/announcementtext long text fields (duties, qualifications)

FIXES in this version
---------------------
1. Pagination bug (cause of the 503s). The API returns `continuationToken`
   ALREADY URL-encoded (e.g. "...%3D%3D"). The previous version passed it
   through urlencode() again, sending "...%253D%253D" - a token that does not
   exist - so page 1 of every series-year succeeded and page 2 always failed
   with 503. We now follow the API's own `paging.next` link verbatim, which is
   correctly encoded AND carries the original filters. If `next` is absent we
   fall back to the decoded token plus the original filters.
2. Nothing fetched is ever thrown away. Each page is appended to a `.partial`
   file and the resume point is checkpointed, so an interrupted walk continues
   where it stopped instead of restarting. The file is renamed to its final
   name only when the walk completes.
3. Colab/Jupyter safe: uses parse_known_args(), so the kernel's `-f` argument
   no longer crashes argparse.
4. A completion report at the end shows exactly which of the expected files
   exist, are partial, or are missing.

Usage:
    python src/06_usajobs_historic.py --start 2019 --end 2025
    python src/06_usajobs_historic.py --skip-text          # structured only, faster
    python src/06_usajobs_historic.py --build-only         # assemble panel only
    python src/06_usajobs_historic.py --status             # completion report only
"""
import argparse
import json
import os
import pathlib
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://data.usajobs.gov"
JOA = "/api/historicjoa"
TEXT = "/api/historicjoa/announcementtext"
UA = "PipelineParadox-DBA-research/1.0"

DEFAULT_SERIES = [
    "2210",  # information technology management
    "1550",  # computer science
    "1560",  # data science
    "0343",  # management and program analysis
    "0110",  # economist
    "0510",  # accounting
    "0905",  # general attorney
    "0801",  # general engineering
    "1515",  # operations research
    "0301",  # miscellaneous administration and program
]

ENTRY_MAX, SENIOR_MIN = 9, 13  # GS grade -> seniority tier (proposal Section 6.3)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def http_get(url: str, retries: int = 6) -> dict:
    """GET a fully-formed URL. Never re-encodes: callers pass final URLs."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            # The API answers with an EMPTY body (not an empty list) when a query
            # matches nothing - e.g. series 1560, which OPM only created in
            # December 2021, has no postings in 2019-2021. Treat that as zero
            # records rather than a failure.
            if not body.strip():
                return {"data": [], "paging": {}}
            return json.loads(body)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = min(120, 5 * 2 ** attempt) + random.uniform(0, 3)
                print(f"      HTTP {e.code}; retry {attempt + 1}/{retries - 1} "
                      f"in {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            if attempt < retries - 1:
                wait = min(120, 5 * 2 ** attempt) + random.uniform(0, 3)
                print(f"      network error ({e}); retry in {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"exhausted retries: {last}")


def first_url(path: str, params: dict) -> str:
    return f"{BASE}{path}?{urllib.parse.urlencode(params)}"


def next_url(path: str, payload: dict, params: dict) -> str | None:
    """Build the URL for the next page, or None when the dataset is exhausted.

    Preferred: the API's own `paging.next` link - already correctly encoded
    and carrying the original filters. Fallback: decode the token once, then
    encode it exactly once alongside the original filters.
    """
    paging = payload.get("paging") or {}
    meta = paging.get("metadata") or {}
    token = meta.get("continuationToken")
    if not token:
        return None
    nxt = paging.get("next")
    if nxt:
        return nxt if nxt.startswith("http") else f"{BASE}{nxt}"
    raw = urllib.parse.unquote(token)                   # undo the API's encoding
    q = dict(params, continuationtoken=raw)             # filters kept
    return f"{BASE}{path}?{urllib.parse.urlencode(q)}"  # encoded exactly once


# ---------------------------------------------------------------------------
# Resumable walk
# ---------------------------------------------------------------------------
def walk(path: str, params: dict, dest: pathlib.Path, pause: float, label: str) -> int:
    """Walk all pages into dest. Resumable; returns total records written."""
    partial = dest.with_suffix(".partial.jsonl")
    ckpt = dest.with_suffix(".ckpt.json")

    if ckpt.exists() and partial.exists():
        state = json.loads(ckpt.read_text())
        url, written, page = state["next_url"], state["written"], state["page"]
        print(f"    resuming {label} at page {page + 1} ({written} already saved)")
    else:
        partial.unlink(missing_ok=True)
        url, written, page = first_url(path, params), 0, 0

    while url:
        payload = http_get(url)
        data = payload.get("data") or []
        with partial.open("a") as fh:
            for rec in data:
                fh.write(json.dumps(rec) + "\n")
        written += len(data)
        page += 1

        total = ((payload.get("paging") or {}).get("metadata") or {}).get("totalCount")
        print(f"    {label} page {page}: +{len(data)} (saved {written}"
              + (f", ~{total} remaining" if total is not None else "") + ")")

        url = next_url(path, payload, params) if data else None
        if url:
            ckpt.write_text(json.dumps({"next_url": url, "written": written, "page": page}))
            time.sleep(pause)

    partial.replace(dest)             # atomic: final name only when complete
    ckpt.unlink(missing_ok=True)
    return written


def year_windows(start: int, end: int) -> list:
    return [(f"{y}-01-01", f"{y}-12-31") for y in range(start, end + 1)]


def fetch(kind: str, series: list, start: int, end: int,
          outdir: pathlib.Path, pause: float) -> dict:
    path = JOA if kind == "joa" else TEXT
    results = {"done": 0, "cached": 0, "failed": []}
    for s in series:
        for w_start, w_end in year_windows(start, end):
            yr = w_start[:4]
            dest = outdir / f"{kind}_{s}_{yr}.jsonl"
            if dest.exists():
                results["cached"] += 1
                print(f"  [cached] {dest.name}")
                continue
            params = {"PositionSeries": s,
                      "StartPositionOpenDate": w_start,
                      "EndPositionOpenDate": w_end}
            print(f"  {kind} series {s} {yr}")
            try:
                n = walk(path, params, dest, pause, f"{s}/{yr}")
                results["done"] += 1
                print(f"    -> {dest.name} ({n} records)")
            except Exception as exc:  # noqa: BLE001
                results["failed"].append(f"{kind}_{s}_{yr}")
                print(f"    FAILED {kind} {s} {yr}: {type(exc).__name__}: {exc}\n"
                      f"    progress kept in {dest.with_suffix('.partial.jsonl').name}; "
                      f"re-run to resume", file=sys.stderr)
            time.sleep(pause)
    return results


# ---------------------------------------------------------------------------
# Completion report
# ---------------------------------------------------------------------------
def status(series: list, start: int, end: int, outdir: pathlib.Path,
           kinds: tuple) -> None:
    rows = []
    for kind in kinds:
        for s in series:
            for y in range(start, end + 1):
                dest = outdir / f"{kind}_{s}_{y}.jsonl"
                part = dest.with_suffix(".partial.jsonl")
                if dest.exists():
                    rows.append((kind, s, y, "complete"))
                elif part.exists():
                    rows.append((kind, s, y, "PARTIAL"))
                else:
                    rows.append((kind, s, y, "missing"))
    total = len(rows)
    comp = sum(r[3] == "complete" for r in rows)
    part = sum(r[3] == "PARTIAL" for r in rows)
    print(f"\n[status] {comp}/{total} complete | {part} partial | "
          f"{total - comp - part} missing")
    for kind in kinds:
        print(f"\n  {kind:<5} " + "".join(f"{y:>6}" for y in range(start, end + 1)))
        for s in series:
            marks = []
            for y in range(start, end + 1):
                st = next(r[3] for r in rows if r[:3] == (kind, s, y))
                marks.append({"complete": "    ok", "PARTIAL": "    ..",
                              "missing": "    --"}[st])
            print(f"  {s:<5} " + "".join(marks))
    print("\n  ok = complete   .. = partial (re-run to resume)   -- = not started")


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
def grade_tier(rec) -> str | None:
    try:
        lo = int(str(rec.get("minimumGrade")).strip())
    except (TypeError, ValueError):
        return None
    return "entry" if lo <= ENTRY_MAX else "senior" if lo >= SENIOR_MIN else "mid"


def build_panel(outdir: pathlib.Path, out_csv: pathlib.Path) -> None:
    import pandas as pd

    def load(prefix):
        # complete files only - partials are excluded from the panel
        files = [f for f in sorted(outdir.glob(f"{prefix}_*.jsonl"))
                 if ".partial" not in f.name and f.stat().st_size]
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)

    joa, txt = load("joa"), load("text")
    if joa.empty:
        print("[panel] no complete structured files yet", file=sys.stderr)
        return

    joa = joa.drop_duplicates("usajobsControlNumber")
    joa["seniority_tier"] = joa.apply(grade_tier, axis=1)
    joa["series"] = joa["jobCategories"].apply(
        lambda v: (v[0] or {}).get("series") if isinstance(v, list) and v else None)
    joa["state"] = joa["positionLocations"].apply(
        lambda v: (v[0] or {}).get("positionLocationState")
        if isinstance(v, list) and v else None)

    keep = ["usajobsControlNumber", "announcementNumber", "series", "positionTitle",
            "hiringAgencyName", "hiringDepartmentName", "state", "payScale",
            "minimumGrade", "maximumGrade", "promotionPotential", "seniority_tier",
            "minimumSalary", "maximumSalary", "salaryType", "workSchedule",
            "supervisoryStatus", "teleworkEligible", "positionOpenDate",
            "positionCloseDate", "totalOpenings", "whoMayApply"]
    panel = joa[[c for c in keep if c in joa.columns]].copy()

    if not txt.empty:
        txt = txt.drop_duplicates("usajobsControlNumber")
        tcols = ["usajobsControlNumber", "duties", "summary", "majorDutiesList",
                 "requirementsQualifications", "requirementsEducation"]
        panel = panel.merge(txt[[c for c in tcols if c in txt.columns]],
                            on="usajobsControlNumber", how="left")

    # the API reports unknown salaries as 0.0 - treat as missing, not as pay
    for col in ("minimumSalary", "maximumSalary"):
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
            panel.loc[panel[col] <= 0, col] = float("nan")

    panel["positionOpenDate"] = pd.to_datetime(panel["positionOpenDate"], errors="coerce")
    panel["quarter"] = panel["positionOpenDate"].dt.to_period("Q").astype(str)
    panel["post"] = (panel["positionOpenDate"] >= "2022-12-01").astype(int)
    panel["gs_only"] = (panel["payScale"].astype(str).str.upper() == "GS").astype(int)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_csv, index=False)
    print(f"\n[panel] {len(panel):,} unique announcements -> {out_csv}")
    if "duties" in panel.columns:
        print(f"        with duty text: {panel['duties'].notna().sum():,}")
    print(f"        GS pay scale:   {int(panel['gs_only'].sum()):,}")
    print(panel["seniority_tier"].value_counts(dropna=False).to_string())
    print("\n[panel] REMINDER: report federal-representativeness caveat with every result.")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", default=DEFAULT_SERIES)
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--outdir", default="data/raw/usajobs")
    ap.add_argument("--panel", default="data/interim/usajobs_panel.csv")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between calls")
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--status", action="store_true")
    args, _unknown = ap.parse_known_args(argv)   # tolerate Jupyter's -f argument

    try:
        os.getcwd()
        cwd_ok = os.path.isdir(".")
    except (FileNotFoundError, OSError):
        cwd_ok = False
    if not cwd_ok:
        print("[usajobs] The notebook's working folder no longer exists - Google Drive\n"
              "          disconnected, or the project folder was moved/replaced.\n"
              "          Run the 'Reconnect' cell (remount Drive + chdir to the project),\n"
              "          then re-run this cell. Completed and partial downloads on Drive\n"
              "          are kept and will be skipped/resumed.", file=sys.stderr)
        return 1

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kinds = ("joa",) if args.skip_text else ("joa", "text")

    if args.status:
        status(args.series, args.start, args.end, outdir, kinds)
        return 0

    if not args.build_only:
        n_expected = len(args.series) * (args.end - args.start + 1)
        print(f"[usajobs] {len(args.series)} series x {args.end - args.start + 1} years "
              f"= {n_expected} files per endpoint; no API key required")
        for kind in kinds:
            r = fetch(kind, args.series, args.start, args.end, outdir, args.pause)
            print(f"\n[usajobs] {kind}: {r['done']} fetched, {r['cached']} cached, "
                  f"{len(r['failed'])} failed")
        status(args.series, args.start, args.end, outdir, kinds)

    build_panel(outdir, pathlib.Path(args.panel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
