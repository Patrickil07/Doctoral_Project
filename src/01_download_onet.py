"""
01_download_onet.py — fetch and unpack a pinned O*NET database release.

O*NET is a free public download. This is NOT scraping: it retrieves the
published database archive from the O*NET Resource Center and records the
release version so the analysis is reproducible.

Usage:
    python src/01_download_onet.py --release 30.3
    python src/01_download_onet.py --release 28.0 --outdir data/raw/onet_28_0

Pin ONE release for the main specification (proposal: "hold task
composition fixed at a single pre-period O*NET release"): the pipeline uses
27.0 (August 2022, before ChatGPT). 30.3 is downloaded too, for the robustness
check and as the reference for GWA names (step 02 --id-reference).
"""
import argparse
import hashlib
import io
import json
import pathlib
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone

BASE = "https://www.onetcenter.org/dl_files/database"
UA = "PipelineParadox-DBA-research/1.0 (academic use)"
# SHA-256 of each release archive as first downloaded (GitHub run, 27 Sep 2026).
KNOWN_SHA256 = {
    "30.3": "7758ec966fd91895b3d290b83c9f1f1d46730d37fdda4faac67104d1c0d2a780",
}


def build_url(release: str) -> str:
    # O*NET publishes text archives as db_<major>_<minor>_text.zip
    tag = release.replace(".", "_")
    return f"{BASE}/db_{tag}_text.zip"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, help="e.g. 30.3")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--url", default=None, help="override if the file name pattern changes")
    args = ap.parse_args()

    url = args.url or build_url(args.release)
    outdir = pathlib.Path(args.outdir or f"data/raw/onet_{args.release.replace('.', '_')}")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[onet] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            blob = resp.read()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[onet] FAILED: {exc}\n"
            "       If the file name pattern has changed, open\n"
            "       https://www.onetcenter.org/database.html, copy the text-archive\n"
            "       link for your release, and re-run with --url <link>.",
            file=sys.stderr,
        )
        return 1

    sha = hashlib.sha256(blob).hexdigest()
    expected = KNOWN_SHA256.get(args.release)
    if expected and not args.url and sha != expected:
        print(f"[onet] FAILED: checksum {sha} does not match the pinned {expected} for "
              f"release {args.release}; nothing extracted", file=sys.stderr)
        return 1
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        members = zf.namelist()
        zf.extractall(outdir)

    manifest = {
        "release": args.release,
        "url": url,
        "sha256": sha,
        "bytes": len(blob),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(members),
    }
    (outdir / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[onet] extracted {len(members)} files to {outdir}")
    print(f"[onet] sha256={sha}")
    print("[onet] cite this release and sha256 in the methodology appendix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
