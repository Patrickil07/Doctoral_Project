"""
02_build_task_composition.py — occupation-level four-part task composition + ILR.

Implements the task-composition measure of the proposal (methods chapter):
  - aggregate O*NET Work Activities IMPORTANCE ratings into c1..c4
  - close to the simplex S^4
  - multiplicative zero replacement (Martin-Fernandez et al. 2003)
  - ILR balances z1,z2,z3 under the pre-specified sequential binary partition
  - ln T scale control

Aggregation (--aggregate). The default, `mean`, gives each part the MEAN
importance of its work activities (GWAs). Every occupation is rated on every
GWA and importance is at least 1, so the older `sum` rule made each share
mostly a count of how many GWAs the mapping puts in the part (12 / 13 / 8 / 8),
not what the occupation does. `sum` is kept for the robustness check. ln T is
the log of the summed importance under either rule.

Zero replacement never triggers on importance ratings (they are at least 1);
the log says so, and --delta cannot change the results.

Release (--id-reference). The mapping names GWAs as O*NET 30.3 does. Older
releases name some GWAs differently, so for any other release pass the 30.3
folder here: mapping names are translated to O*NET Element IDs (stable across
releases) in 30.3 and matched on Element ID in the release being built.

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


def read_work_activities(onet_dir: pathlib.Path) -> pd.DataFrame | None:
    wa = pd.read_csv(find_file(onet_dir, "Work Activities"), sep="\t", dtype=str)
    wa.columns = [c.strip() for c in wa.columns]
    needed = {"O*NET-SOC Code", "Element ID", "Element Name", "Scale ID", "Data Value"}
    missing = needed - set(wa.columns)
    if missing:
        print(f"[task] unexpected Work Activities schema in {onet_dir}, missing {missing}",
              file=sys.stderr)
        return None
    wa["Data Value"] = pd.to_numeric(wa["Data Value"], errors="coerce")
    return wa


def aggregate_parts(df: pd.DataFrame, how: str) -> tuple[pd.DataFrame, pd.Series]:
    """Part scores per O*NET-SOC occupation (mean or sum of GWA ratings) and the
    total rating T (always the sum over all mapped GWAs)."""
    g = df.groupby(["O*NET-SOC Code", "task_part"])["Data Value"]
    agg = (g.mean() if how == "mean" else g.sum()).unstack("task_part")
    agg = agg.reindex(columns=PARTS).fillna(0.0)
    T = df.groupby("O*NET-SOC Code")["Data Value"].sum().reindex(agg.index)
    return agg, T


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
    ap.add_argument("--aggregate", default="mean", choices=["mean", "sum"],
                    help="mean (default) or sum of GWA ratings within each part")
    ap.add_argument("--id-reference", default=None,
                    help="O*NET release folder whose GWA names the mapping uses; "
                         "match the target release on Element ID through it")
    args = ap.parse_args()

    onet_dir = pathlib.Path(args.onet)
    wa = read_work_activities(onet_dir)
    if wa is None:
        return 1
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
    if args.id_reference:
        ref = read_work_activities(pathlib.Path(args.id_reference))
        if ref is None:
            return 1
        ids = (ref.assign(key=norm(ref["Element Name"]))
                  .drop_duplicates("key").set_index("key")["Element ID"])
        missing_ref = sorted(set(mp["key"]) - set(ids.index))
        if missing_ref:
            print(f"[task] MAPPING ERROR — not in the reference release: {missing_ref}",
                  file=sys.stderr)
            return 2
        mp["key"] = mp["key"].map(ids).str.strip()
        wa["key"] = wa["Element ID"].str.strip()
        renamed = (wa.drop_duplicates("key").set_index("key")["Element Name"]
                     .reindex(mp["key"]))
        for old, new in zip(mp["element_name"], renamed):
            if isinstance(new, str) and norm(pd.Series([new]))[0] != norm(pd.Series([old]))[0]:
                print(f"[task] matched by Element ID: '{old}' is '{new}' in this release")

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
    agg, T = aggregate_parts(df, args.aggregate)
    print(f"[task] part score = {args.aggregate} of GWA {args.scale} ratings; GWAs per "
          f"part: {mp['task_part'].value_counts().reindex(PARTS).tolist()}")
    keep = T > 0
    if (~keep).any():
        print(f"[task] dropping {(~keep).sum()} occupations with zero total weight")
    agg, T = agg[keep], T[keep]

    n_zero = int((agg.to_numpy() == 0).sum())
    print(f"[task] zero part scores: {n_zero}"
          + (" (zero replacement not used; --delta has no effect)" if not n_zero else ""))
    C = multiplicative_replacement(agg.to_numpy(), args.delta)
    Z = ilr_balances(C)

    out = pd.DataFrame(C, columns=PARTS, index=agg.index)
    out[["z1", "z2", "z3"]] = Z
    out["ln_T"] = np.log(T.to_numpy())
    out["onet_soc"] = out.index
    out["soc2018"] = out["onet_soc"].str.slice(0, 7)  # 12-3456 detailed SOC
    out["onet_release_dir"] = str(onet_dir)
    out["zero_delta"] = args.delta
    out["aggregate"] = args.aggregate

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
