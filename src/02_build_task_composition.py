"""
02_build_task_composition.py — occupation-level four-part task composition + ILR.

Implements proposal Sections 6.4 / 3.4:
  - aggregate O*NET Work Activities IMPORTANCE ratings into c1..c4
  - close to the simplex S^4
  - multiplicative zero replacement (Martin-Fernandez et al. 2003)
  - ILR balances z1,z2,z3 under the pre-specified sequential binary partition
  - ln T scale control

Fails loudly if any element name in the mapping file is absent from the
downloaded O*NET release, so a silent partial mapping can never occur.

Usage:
    python src/02_build_task_composition.py
        --onet data/raw/onet_30_3
        --map mapping/onet_activity_map.csv
        --out data/interim/task_composition.csv
"""
import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd

PARTS = ["c1_nonroutine_analytic", "c2_nonroutine_interpersonal",
         "c3_routine_cognitive", "c4_residual"]


def norm(s: pd.Series) -> pd.Series:
    """Normalise element names so punctuation differences don't break matching."""
    return (s.astype(str)
             .str.replace(r"[^\w\s]", " ", regex=True)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip()
             .str.lower())


def find_file(onet_dir: pathlib.Path, stem: str) -> pathlib.Path:
    hits = [p for p in onet_dir.rglob("*.txt") if p.stem.lower() == stem.lower()]
    if not hits:
        raise FileNotFoundError(f"{stem}.txt not found under {onet_dir}")
    return hits[0]


def multiplicative_replacement(X: np.ndarray, delta: float) -> np.ndarray:
    """Martin-Fernandez et al. (2003) multiplicative zero replacement."""
    X = X.astype(float).copy()
    out = np.empty_like(X)
    for i, row in enumerate(X):
        zeros = row == 0
        if not zeros.any():
            out[i] = row / row.sum()
            continue
        r = row / row.sum()
        n_zero = zeros.sum()
        r[zeros] = delta
        r[~zeros] = r[~zeros] * (1 - n_zero * delta)
        out[i] = r / r.sum()
    return out


def ilr_balances(C: np.ndarray) -> np.ndarray:
    """
    Sequential binary partition (proposal Eq. 2-4):
        z1: {c1,c2,c3} vs {c4}
        z2: {c1,c3}    vs {c2}
        z3: {c1}       vs {c3}
    Balance formula: sqrt(rs/(r+s)) * ln( g(num)/g(den) )
    """
    c1, c2, c3, c4 = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
    g123 = np.cbrt(c1 * c2 * c3)
    g13 = np.sqrt(c1 * c3)
    z1 = np.sqrt(3.0 / 4.0) * np.log(g123 / c4)
    z2 = np.sqrt(2.0 / 3.0) * np.log(g13 / c2)
    z3 = np.sqrt(1.0 / 2.0) * np.log(c1 / c3)
    return np.column_stack([z1, z2, z3])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onet", required=True)
    ap.add_argument("--map", default="mapping/onet_activity_map.csv")
    ap.add_argument("--out", default="data/interim/task_composition.csv")
    ap.add_argument("--delta", type=float, default=1e-4,
                    help="zero-replacement value; vary in sensitivity analysis")
    ap.add_argument("--scale", default="IM", choices=["IM", "LV"],
                    help="IM = Importance (default), LV = Level")
    args = ap.parse_args()

    onet_dir = pathlib.Path(args.onet)
    wa = pd.read_csv(find_file(onet_dir, "Work Activities"), sep="\t", dtype=str)
    wa.columns = [c.strip() for c in wa.columns]

    needed = {"O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"}
    missing = needed - set(wa.columns)
    if missing:
        print(f"[task] unexpected Work Activities schema, missing {missing}", file=sys.stderr)
        return 1

    wa["Data Value"] = pd.to_numeric(wa["Data Value"], errors="coerce")
    wa = wa[wa["Scale ID"].str.strip() == args.scale].dropna(subset=["Data Value"])

    # The mapping is a methodological choice and lives under version control;
    # it is never generated or overwritten here.
    map_path = pathlib.Path(args.map)
    if not map_path.exists():
        print(f"[task] mapping file {map_path} not found", file=sys.stderr)
        return 1
    mp = pd.read_csv(map_path)
    dup = mp["element_name"][mp["element_name"].duplicated()]
    if not dup.empty:
        print(f"[task] MAPPING ERROR — elements assigned more than once: {sorted(dup)}",
              file=sys.stderr)
        return 2
    mp["key"] = norm(mp["element_name"])
    wa["key"] = norm(wa["Element Name"])

    onet_keys = set(wa["key"])
    map_keys = set(mp["key"])

    unmatched_map = sorted(map_keys - onet_keys)
    if unmatched_map:
        print("[task] MAPPING ERROR — these mapping entries match no O*NET element:",
              file=sys.stderr)
        for k in unmatched_map:
            print(f"        - {k}", file=sys.stderr)
        print("       Fix mapping/onet_activity_map.csv against this release "
              "before proceeding. Do not run the analysis on a partial mapping.",
              file=sys.stderr)
        return 2

    unmapped_onet = sorted(onet_keys - map_keys)
    if unmapped_onet:
        print(f"[task] WARNING — {len(unmapped_onet)} O*NET activities are unmapped "
              f"and will be dropped:")
        for k in unmapped_onet:
            print(f"        - {k}")
        print("       Every Work Activity should be assigned to exactly one part for "
              "the composition to be exhaustive. Assign these before final estimation.")

    df = wa.merge(mp[["key", "task_part"]], on="key", how="inner")
    agg = (df.groupby(["O*NET-SOC Code", "task_part"])["Data Value"]
             .sum().unstack("task_part").reindex(columns=PARTS).fillna(0.0))

    T = agg.sum(axis=1)
    keep = T > 0
    if (~keep).any():
        print(f"[task] dropping {(~keep).sum()} occupations with zero total weight")
    agg, T = agg[keep], T[keep]

    C = multiplicative_replacement(agg.to_numpy(), args.delta)
    Z = ilr_balances(C)

    out = pd.DataFrame(C, columns=PARTS, index=agg.index)
    out[["z1", "z2", "z3"]] = Z
    out["ln_T"] = np.log(T.to_numpy())
    out["onet_soc"] = out.index
    out["soc2018"] = out["onet_soc"].str.slice(0, 7)  # 12-3456 detailed SOC
    out["onet_release_dir"] = str(onet_dir)
    out["zero_delta"] = args.delta

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.reset_index(drop=True).to_csv(args.out, index=False)

    print(f"[task] {len(out)} O*NET-SOC occupations written to {args.out}")
    print("[task] composition means:")
    print(out[PARTS].mean().round(4).to_string())
    print(f"[task] z3 (analytic:routine balance) mean={out['z3'].mean():.4f} "
          f"sd={out['z3'].std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
