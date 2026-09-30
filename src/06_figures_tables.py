"""
06_figures_tables.py — Chapter 4 figures and formatted tables from saved results.

Reads only aggregate outputs that steps 03, 05 and 07 already wrote (result
CSVs, occupation measures, the step 05 sample log). It never opens microdata
or the analysis sample, so it can run on any saved run folder, e.g. one from
the `results` branch:

    python src/06_figures_tables.py --root runs/2026-09-28_run5_8d9019c \
        --out runs/2026-09-28_run5_8d9019c/chapter4

Specifications are discovered, not hard-coded: every folder
<root>/data/out/results*/ written by step 07 is one specification. The main
one is `results`; to add a specification (e.g. an extra RQ1 variant), run
step 07 with `--out data/out/results_<name>` and it appears in every table and
comparison figure. Give it a readable name with `--label results_<name>="..."`
or in SPEC_LABELS below.

Outputs (in --out):
  figures/  fig_*.png (300 dpi) and fig_*.pdf
  tables/   table_*.csv (unrounded), table_*.md, table_*.tex (booktabs)
  chapter4_tables.xlsx   every table, one sheet each (for pasting into Word)
  SOURCES.txt            which run and which code produced these files
"""
import argparse
import os
import pathlib
import re
import subprocess

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REF_Q = "2022Q3"                 # must equal REF_Q in 07_estimate.py (tested)
KNOWLEDGE_MAJOR = {"13", "15", "17", "19", "23", "27", "43"}  # as in 05 (tested)

MAIN = "results"
# Readable names for the specifications step 07 writes; order = display order.
SPEC_LABELS = {
    "results": "Main specification",
    "results_drop_pandemic": "Pandemic window dropped",
    "results_exposure_gpt4beta": "GPT-4-rated exposure",
    "results_exposure_lmaioe": "LM-AIOE exposure",
    "results_rq1_trend": "RQ1 Model 2: linear trend",
    "results_rq1_from_2021q4": "RQ1 Model 3: from 2021Q4",
}

# Event-study outcomes: file in each specification folder -> figure/table meta.
EVENT_STUDIES = {
    "rq1": ("rq1_task_composition.csv",
            "RQ1: task composition (z3) of early-career employment",
            "Exposure × early-career coefficient on z3"),
    "rq2": ("rq2_early_share.csv",
            "RQ2: early-career employment share",
            "Exposure coefficient on early-career share"),
    "rq4_uncond": ("rq4_earnings_unconditional.csv",
                   "RQ4: log weekly earnings, no task controls",
                   "Exposure × early-career coefficient on ln W"),
    "rq4_cond": ("rq4_earnings_conditional.csv",
                 "RQ4: log weekly earnings, with task controls",
                 "Exposure × early-career coefficient on ln W"),
}

RQ3_TERMS = {
    "z1": "θ1  z1 (analytic vs rest)",
    "z2": "θ2  z2",
    "z3": "θ3  z3",
    "ln_T": "λ  ln T",
    "z1:early_career": "θJ1  z1 × early",
    "z2:early_career": "θJ2  z2 × early",
    "z3:early_career": "θJ3  z3 × early",
    "z1:early_career:post": "ψ1  z1 × early × post",
    "z2:early_career:post": "ψ2  z2 × early × post",
    "z3:early_career:post": "ψ3  z3 × early × post",
    "z1:post": "π1  z1 × post",
    "z2:post": "π2  z2 × post",
    "z3:post": "π3  z3 × post",
    "ln_T:post": "λP  ln T × post",
    "early_career:post": "ρ  early × post",
}

# Categorical palette (validated for CVD separation); every series also gets
# its own marker so identity never rests on colour alone (print, greyscale).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#dddddd"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.5, "legend.frameon": False, "savefig.dpi": 300,
    "figure.dpi": 100, "pdf.fonttype": 42,
})


# --- loading -----------------------------------------------------------------
def label_for(spec: str, labels: dict) -> str:
    if spec in labels:
        return labels[spec]
    return spec.removeprefix("results_").replace("_", " ").capitalize()


def discover_specs(root: pathlib.Path, only: list | None = None) -> list:
    """Specification folders under <root>/data/out, main first, then SPEC_LABELS
    order, then any others alphabetically."""
    found = sorted(p.name for p in (root / "data" / "out").glob("results*")
                   if p.is_dir() and any(p.glob("*.csv")))
    if only:
        missing = [s for s in only if s not in found]
        if missing:
            raise SystemExit(f"[06] specification folder(s) not found: {missing}")
        return only
    known = [s for s in SPEC_LABELS if s in found]
    return known + [s for s in found if s not in known]


def read_result(root: pathlib.Path, spec: str, fname: str) -> pd.DataFrame | None:
    p = root / "data" / "out" / spec / fname
    return pd.read_csv(p, dtype={"quarter": str}) if p.exists() else None


def with_reference(es: pd.DataFrame) -> pd.DataFrame:
    """Add the omitted reference quarter as a zero point (no standard error)."""
    if REF_Q in set(es["quarter"]):
        return es
    ref = pd.DataFrame({"term": [f"ref_{REF_Q}"], "estimate": [0.0], "se": [0.0],
                        "pvalue": [np.nan], "quarter": [REF_Q], "period": ["ref"]})
    return pd.concat([es, ref], ignore_index=True).sort_values("quarter").reset_index(drop=True)


def parse_sample_log(path: pathlib.Path) -> pd.DataFrame | None:
    """Table 4.1 rows from the step 05 log (the block printed under
    'SAMPLE CONSTRUCTION LOG'), plus the early-career / experienced split."""
    if not path.exists():
        return None
    rows, inside = [], False
    text = path.read_text()
    for line in text.splitlines():
        if "SAMPLE CONSTRUCTION LOG" in line:
            inside = True
            continue
        if inside:
            m = re.match(r"^\s+(\S.*?)\s{2,}([\d,]+)\s*$", line)
            if m:
                rows.append((m.group(1), int(m.group(2).replace(",", ""))))
            elif rows:
                break
    if not rows:
        return None
    t = pd.DataFrame(rows, columns=["step", "records"])
    t["dropped"] = (-t["records"].diff()).astype("Int64")
    m = re.search(r"early-career:\s*([\d,]+)\s*\|\s*experienced:\s*([\d,]+)", text)
    if m:
        extra = pd.DataFrame({
            "step": ["  of which early-career (22-30)", "  of which experienced (35-55)"],
            "records": [int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))],
            "dropped": pd.array([pd.NA, pd.NA], dtype="Int64")})
        t = pd.concat([t, extra], ignore_index=True)
    return t


# --- formatting ----------------------------------------------------------------
def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def coef_cells(est: float, se: float, p: float, digits: int = 3) -> tuple[str, str]:
    if pd.isna(est):
        return "", ""
    return f"{est:.{digits}f}{stars(p)}", f"({se:.{digits}f})"


def stacked_coef_table(blocks: dict, keys: list, key_labels: dict | None = None) -> pd.DataFrame:
    """Rows: each key's estimate, then its SE in parentheses; columns: blocks
    (specifications). `blocks` maps column label -> DataFrame indexed by key."""
    key_labels = key_labels or {}
    out = []
    for k in keys:
        est_row = {"": key_labels.get(k, k)}
        se_row = {"": ""}
        for col, df in blocks.items():
            if k in df.index:
                r = df.loc[k]
                est_row[col], se_row[col] = coef_cells(r["estimate"], r["se"], r["pvalue"])
            else:
                est_row[col] = se_row[col] = ""
        out += [est_row, se_row]
    return pd.DataFrame(out)


def to_markdown(df: pd.DataFrame, note: str = "") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v).replace("|", "\\|")
                                       for v in r) + " |")
    return "\n".join(lines) + (f"\n\n{note}\n" if note else "\n")


def _tex_escape(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("×", r"$\times$"), ("ψ", r"$\psi$"), ("θ", r"$\theta$"),
                 ("λ", r"$\lambda$"), ("ρ", r"$\rho$"), ("|", r"$|$")]:
        s = s.replace(a, b)
    return re.sub(r"(\*+)$", lambda m: "$^{" + m.group(1) + "}$", s)


def to_latex(df: pd.DataFrame, caption: str, note: str = "") -> str:
    cols = [str(c) for c in df.columns]
    body = [r"\begin{table}[htbp]\centering", rf"\caption{{{_tex_escape(caption)}}}",
            r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
            " & ".join(_tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        body.append(" & ".join("" if pd.isna(v) else _tex_escape(str(v)) for v in r) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    if note:
        body.append(r"\par\vspace{2pt}{\footnotesize " + _tex_escape(note) + "}")
    body.append(r"\end{table}")
    return "\n".join(body) + "\n"


class TableWriter:
    def __init__(self, outdir: pathlib.Path):
        self.dir = outdir / "tables"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sheets = {}

    def write(self, name: str, formatted: pd.DataFrame, caption: str,
              note: str = "", raw: pd.DataFrame | None = None):
        (raw if raw is not None else formatted).to_csv(self.dir / f"{name}.csv", index=False)
        (self.dir / f"{name}.md").write_text(f"**{caption}**\n\n" + to_markdown(formatted, note))
        (self.dir / f"{name}.tex").write_text(to_latex(formatted, caption, note))
        self.sheets[name] = (formatted, caption, note)
        print(f"[06] table  {name}")

    def workbook(self, path: pathlib.Path):
        if not self.sheets:
            return
        with pd.ExcelWriter(path) as xw:
            for name, (df, caption, note) in self.sheets.items():
                sheet = name.removeprefix("table_")[:31]
                df.to_excel(xw, sheet_name=sheet, index=False, startrow=2)
                ws = xw.sheets[sheet]
                ws.cell(row=1, column=1, value=caption)
                if note:
                    ws.cell(row=len(df) + 5, column=1, value=note)
        print(f"[06] workbook {path.name}")


def save_fig(fig, outdir: pathlib.Path, name: str):
    d = outdir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[06] figure {name}")


# --- figures -------------------------------------------------------------------
def _quarter_axis(ax, quarters: list):
    x = np.arange(len(quarters))
    ax.set_xticks(x)
    ax.set_xticklabels([q[:4] if q.endswith("Q1") else "" for q in quarters])
    ax.set_xlim(-0.6, len(quarters) - 0.4)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    if REF_Q in quarters:
        ax.axvline(quarters.index(REF_Q) + 0.5, color=MUTED, linewidth=0.8,
                   linestyle="--", zorder=1)
    return x


def plot_event(ax, es: pd.DataFrame, i: int = 0, offset: float = 0.0,
               label: str | None = None, quarters: list | None = None):
    quarters = quarters or event_quarters([es])
    # Reindex on the full quarter axis so dropped quarters (e.g. the pandemic
    # window) show as gaps rather than being bridged by the line.
    es = with_reference(es).set_index("quarter").reindex(quarters)
    x = np.arange(len(quarters)) + offset
    y, ci = es["estimate"].to_numpy(dtype=float), 1.96 * es["se"].to_numpy(dtype=float)
    c = COLORS[i % len(COLORS)]
    ax.errorbar(x, y, yerr=ci, fmt="none", ecolor=c, elinewidth=1.0, capsize=0,
                alpha=0.8, zorder=2)
    ax.plot(x, y, color=c, linewidth=1.2, marker=MARKERS[i % len(MARKERS)],
            markersize=4.5, markerfacecolor=c, markeredgecolor="white",
            markeredgewidth=0.6, label=label, zorder=3)


def event_quarters(frames: list) -> list:
    qs = set()
    for f in frames:
        qs |= set(f["quarter"])
    return sorted(qs | {REF_Q})


def fig_event_main(es: pd.DataFrame, title: str, ylab: str):
    quarters = event_quarters([es])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    _quarter_axis(ax, quarters)
    plot_event(ax, es, 0, quarters=quarters)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylab)
    ax.text(quarters.index(REF_Q) + 0.6, ax.get_ylim()[1], "post period (from 2022Q4) →",
            color=MUTED, fontsize=8, va="top")
    fig.text(0.01, -0.04, f"Points: estimates; bars: 95% CI (SE clustered on occupation). "
             f"Reference quarter {REF_Q} = 0.", color=MUTED, fontsize=7.5)
    return fig


def fig_event_by_spec(frames: dict, title: str, ylab: str):
    """Small multiples, one panel per specification. Each panel keeps its own
    y-axis: exposure measures are on different scales, so their coefficients
    are not comparable in size."""
    n = len(frames)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    quarters = event_quarters(list(frames.values()))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5, 2.3 * nrow + 0.4),
                             sharey=False, squeeze=False)
    for k, (lab, es) in enumerate(frames.items()):
        ax = axes.flat[k]
        _quarter_axis(ax, quarters)
        plot_event(ax, es, k, quarters=quarters)
        ax.set_title(lab, loc="left", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.supylabel(ylab, fontsize=8)
    fig.tight_layout()
    return fig


def fig_rq4(uncond: pd.DataFrame, cond: pd.DataFrame, decomp: pd.DataFrame | None):
    quarters = event_quarters([uncond, cond])
    rows = 2 if decomp is not None else 1
    fig, axes = plt.subplots(rows, 1, figsize=(6.5, 3.2 * rows), sharex=True,
                             squeeze=False, gridspec_kw={"height_ratios": [2, 1][:rows]})
    ax = axes[0, 0]
    _quarter_axis(ax, quarters)
    plot_event(ax, uncond, 0, -0.15, "Without task controls", quarters)
    plot_event(ax, cond, 1, 0.15, "With task controls (z1–z3, ln T)", quarters)
    ax.set_title("RQ4: log weekly earnings, exposure × early-career", loc="left")
    ax.set_ylabel("Coefficient on ln W")
    ax.legend(loc="lower left", fontsize=8)
    if decomp is not None:
        ax2 = axes[1, 0]
        x = _quarter_axis(ax2, quarters)
        d = decomp.set_index("term" if "quarter" not in decomp else "quarter")
        vals = [d["composition_component"].get(q, 0.0) if q != REF_Q else 0.0
                for q in quarters]
        ax2.bar(x, vals, width=0.6, color=COLORS[2], zorder=2)
        ax2.set_ylabel("Composition\n(without − with)")
        ax2.set_title("Difference between the two estimates (point estimate only, no SE)",
                      loc="left", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return fig


def fig_rq3(frames: dict):
    """Coefficient plot of ψ terms (and ρ) across specifications."""
    terms = ["z1:early_career:post", "z2:early_career:post", "z3:early_career:post",
             "early_career:post"]
    n = len(frames)
    fig, ax = plt.subplots(figsize=(6.5, 0.9 + 0.55 * len(terms) * max(n, 2) / 2))
    step = 0.7 / max(n, 1)
    for k, (lab, df) in enumerate(frames.items()):
        d = df.set_index("term")
        ys, xs, cis = [], [], []
        for j, t in enumerate(terms):
            if t in d.index:
                ys.append(j + (k - (n - 1) / 2) * step)
                xs.append(d.loc[t, "estimate"])
                cis.append(1.96 * d.loc[t, "se"])
        c = COLORS[k % len(COLORS)]
        ax.errorbar(xs, ys, xerr=cis, fmt=MARKERS[k % len(MARKERS)], color=c,
                    markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
                    elinewidth=1.0, capsize=0, label=lab)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([RQ3_TERMS[t] for t in terms])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Coefficient on ln W (95% CI)")
    ax.set_title("RQ3: change in early-career task prices after 2022Q3", loc="left")
    if n > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def knowledge_occupations(occ: pd.DataFrame) -> pd.DataFrame:
    major = occ["soc_major"].astype(str).str.zfill(2)
    return occ[major.isin(KNOWLEDGE_MAJOR)].copy()


def fig_exposure_distribution(occ: pd.DataFrame):
    k = knowledge_occupations(occ).dropna(subset=["exposure"])
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.hist(k["exposure"], bins=25, weights=k["emp_total"] / 1e6, color=COLORS[0],
            edgecolor="white", linewidth=0.6, zorder=2)
    ax.set_xlabel("Exposure, main measure (Eloundou et al. human-rated β)")
    ax.set_ylabel("OEWS employment (millions)")
    ax.set_title(f"Exposure across knowledge-intensive occupations (n = {len(k)})",
                 loc="left")
    return fig


def fig_task_composition(occ: pd.DataFrame):
    parts = [("c1_nonroutine_analytic", "Non-routine analytic"),
             ("c2_nonroutine_interpersonal", "Non-routine interpersonal"),
             ("c3_routine_cognitive", "Routine cognitive"),
             ("c4_residual", "Residual")]
    k = knowledge_occupations(occ).dropna(subset=["exposure_tercile"])
    order = [t for t in ("low", "mid", "high") if t in set(k["exposure_tercile"])]
    shares = pd.DataFrame({
        t: [np.average(g[c], weights=g["emp_total"]) for c, _ in parts]
        for t, g in ((t, k[k["exposure_tercile"] == t]) for t in order)},
        index=[lab for _, lab in parts]).T
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    left = np.zeros(len(order))
    for i, (lab, col) in enumerate(shares.items()):
        ax.barh(range(len(order)), col, left=left, color=COLORS[i], height=0.6,
                edgecolor="white", linewidth=1.5, label=lab, zorder=2)
        for y, (l, v) in enumerate(zip(left, col)):
            if v > 0.07:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center",
                        fontsize=7.5, color="white" if i in (0, 5) else INK)
        left += col.to_numpy()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{t.capitalize()} exposure" for t in order])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_title("Task composition by exposure tercile (employment-weighted)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=7.5)
    return fig, shares


# --- tables --------------------------------------------------------------------
STAR_NOTE = ("Standard errors clustered on occupation in parentheses. "
             "* p<0.10, ** p<0.05, *** p<0.01.")


def joint_wald_p(es: pd.DataFrame, vcov: pd.DataFrame | None) -> float:
    """p-value of the Wald test that all pre-period coefficients are zero
    (same test as 07_estimate.joint_wald); NaN without a covariance matrix."""
    from scipy import stats
    terms = list(es.loc[es["period"] == "pre", "term"])
    if vcov is None or not terms or not set(terms) <= set(vcov.index):
        return np.nan
    b = es.set_index("term").loc[terms, "estimate"].to_numpy()
    V = vcov.loc[terms, terms].to_numpy()
    return float(stats.chi2.sf(float(b @ np.linalg.pinv(V) @ b), len(terms)))


def pretrend_rows(es: pd.DataFrame, vcov: pd.DataFrame | None = None) -> dict:
    pre = es[es["period"] == "pre"]
    if pre.empty:
        return {"pre quarters": 0, "max abs z": "", "quarter of max": "", "p<0.05": "",
                "p<0.10": "", "joint p": ""}
    z = (pre["estimate"] / pre["se"]).abs()
    p = joint_wald_p(es, vcov)
    return {"pre quarters": len(pre), "max abs z": round(float(z.max()), 2),
            "quarter of max": pre.loc[z.idxmax(), "quarter"],
            "p<0.05": int((pre["pvalue"] < 0.05).sum()),
            "p<0.10": int((pre["pvalue"] < 0.10).sum()),
            "joint p": "" if np.isnan(p) else f"{p:.3f}"}


def read_vcov(root: pathlib.Path, spec: str, fname: str) -> pd.DataFrame | None:
    p = root / "data" / "out" / spec / fname.replace(".csv", "_vcov.csv")
    return pd.read_csv(p, index_col=0) if p.exists() else None


def fig_honest_did(hd: pd.DataFrame, title: str):
    """Robust 95% sets for the average post coefficient as Mbar grows, with the
    conventional interval on the left."""
    orig = hd[hd["method"] == "Original"].iloc[0]
    rob = hd[hd["method"] != "Original"].sort_values("Mbar")
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    step = float(rob["Mbar"].diff().median()) if len(rob) > 1 else 0.1
    x0 = -2 * step
    ax.errorbar([x0], [(orig["lb"] + orig["ub"]) / 2], yerr=[[(orig["ub"] - orig["lb"]) / 2]],
                fmt="none", ecolor=COLORS[1], elinewidth=2.0, label="Conventional 95% CI")
    ax.vlines(rob["Mbar"], rob["lb"], rob["ub"], color=COLORS[0], linewidth=2.0,
              label="Robust 95% set (relative magnitudes)")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("M̄ (post violations relative to the largest pre-period violation)")
    ax.set_ylabel("Average post-period coefficient")
    ax.set_title(title, loc="left")
    b = hd["breakdown_Mbar"].iloc[0]
    txt = ("breakdown M̄: none (includes 0 at M̄ = 0)" if pd.isna(b) else
           f"breakdown M̄ > {rob['Mbar'].max():g}" if np.isinf(b) else f"breakdown M̄ = {b:g}")
    ax.text(0.99, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
            color=MUTED, fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def build(root: pathlib.Path, outdir: pathlib.Path, specs: list, labels: dict) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    tw = TableWriter(outdir)
    lab = {s: label_for(s, labels) for s in specs}
    print(f"[06] specifications: {', '.join(f'{s} ({lab[s]})' for s in specs)}")

    # Table 4.1 — sample construction
    t41 = parse_sample_log(root / "logs" / "05_sample.log")
    if t41 is not None:
        f = t41.copy()
        f["records"] = f["records"].map("{:,}".format)
        f["dropped"] = f["dropped"].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
        f.columns = ["Step", "Records", "Dropped"]
        tw.write("table_4_1_sample", f, "Table 4.1 Sample construction (CPS basic monthly, 2020-2025)",
                 "Counts from the step 05 sample log.", raw=t41)
    else:
        print("[06] no logs/05_sample.log under root; Table 4.1 skipped")

    # Descriptives from occupation measures (main exposure measure)
    occ_path = root / "data" / "interim" / "occ_measures.csv"
    if occ_path.exists():
        occ = pd.read_csv(occ_path, dtype={"soc_major": str})
        save_fig(fig_exposure_distribution(occ), outdir, "fig_exposure_distribution")
        fig, shares = fig_task_composition(occ)
        save_fig(fig, outdir, "fig_task_composition_by_tercile")
        k = knowledge_occupations(occ)
        rows = []
        for t in ("low", "mid", "high"):
            g = k[k["exposure_tercile"] == t]
            if g.empty:
                continue
            w = g["emp_total"]
            rows.append({"Exposure tercile": t.capitalize(), "Occupations": len(g),
                         "Employment (m)": w.sum() / 1e6,
                         **{c: np.average(g[c], weights=w) for c in
                            ["exposure", "c1_nonroutine_analytic", "c2_nonroutine_interpersonal",
                             "c3_routine_cognitive", "c4_residual", "z1", "z2", "z3", "ln_T"]}})
        raw = pd.DataFrame(rows)
        f = raw.copy()
        f["Employment (m)"] = f["Employment (m)"].map("{:.2f}".format)
        for c in raw.columns[3:]:
            f[c] = f[c].map("{:.3f}".format)
        f = f.rename(columns={"exposure": "Exposure", "c1_nonroutine_analytic": "c1 NR analytic",
                              "c2_nonroutine_interpersonal": "c2 NR interpersonal",
                              "c3_routine_cognitive": "c3 routine cognitive",
                              "c4_residual": "c4 residual", "ln_T": "ln T"})
        tw.write("table_4_2_descriptives", f,
                 "Table 4.2 Occupation measures by exposure tercile (knowledge-intensive occupations)",
                 "Means weighted by OEWS employment. Terciles as assigned in step 03.", raw=raw)
    else:
        print(f"[06] {occ_path} not found; descriptive figures skipped")

    # Event studies: RQ1, RQ2, RQ4
    pre_rows = []
    for key, (fname, title, ylab) in EVENT_STUDIES.items():
        full = {s: read_result(root, s, fname) for s in specs}
        full = {s: d for s, d in full.items() if d is not None}
        if not full:
            continue
        # Model 2 adds a trend row (period "trend", no quarter): tabulated, not plotted
        frames = {s: d[d["period"] != "trend"].reset_index(drop=True) for s, d in full.items()}
        if MAIN in frames and key != "rq4_cond" and key != "rq4_uncond":
            save_fig(fig_event_main(frames[MAIN], title, ylab), outdir, f"fig_{key}_event_study")
        if len(frames) > 1:
            save_fig(fig_event_by_spec({lab[s]: d for s, d in frames.items()}, title, ylab),
                     outdir, f"fig_{key}_event_study_by_spec")
        quarters = sorted(set().union(*(set(d["quarter"]) for d in frames.values())))
        blocks = {lab[s]: d.assign(quarter=d["quarter"].where(d["period"] != "trend", "trend"))
                  .set_index("quarter") for s, d in full.items()}
        has_trend = any((d["period"] == "trend").any() for d in full.values())
        f = stacked_coef_table(blocks, quarters + (["trend"] if has_trend else []),
                               {"trend": "Linear trend E × J × t"})
        f.columns = ["Quarter"] + list(f.columns[1:])
        raw = pd.concat([d.assign(specification=s) for s, d in full.items()], ignore_index=True)
        trend_note = (" Model 2 replaces the pre-period dummies with the linear trend; its "
                      "post coefficients are deviations from the extrapolated trend."
                      if has_trend else "")
        tw.write(f"table_{key}_event_study", f, title,
                 f"Reference quarter {REF_Q} omitted. " + STAR_NOTE + trend_note, raw=raw)
        for s, d in frames.items():
            pre_rows.append({"Outcome": title.split(":")[0] + (
                " (no task controls)" if key == "rq4_uncond" else
                " (task controls)" if key == "rq4_cond" else ""),
                "Specification": lab[s], **pretrend_rows(d, read_vcov(root, s, fname))})

    # Honest DiD (step 08) for RQ1
    hd_rows = []
    for s in specs:
        hd = read_result(root, s, "rq1_task_composition_honest_did.csv")
        if hd is None:
            continue
        save_fig(fig_honest_did(hd, f"RQ1 sensitivity to parallel-trends violations: {lab[s]}"),
                 outdir, f"fig_rq1_honest_did_{s.removeprefix('results').strip('_') or 'main'}")
        for _, r in hd.iterrows():
            hd_rows.append({"Specification": lab[s],
                            "M̄": "conventional" if r["method"] == "Original" else f"{r['Mbar']:g}",
                            "95% set": f"[{r['lb']:.3f}, {r['ub']:.3f}]" + (
                                " (open-ended)" if bool(r.get("at_grid_edge", False)) else ""),
                            "Excludes 0": "yes" if r["excludes_zero"] else "no"})
    if hd_rows:
        raw = pd.concat([read_result(root, s, "rq1_task_composition_honest_did.csv")
                         .assign(specification=s) for s in specs
                         if read_result(root, s, "rq1_task_composition_honest_did.csv") is not None],
                        ignore_index=True)
        tw.write("table_rq1_honest_did", pd.DataFrame(hd_rows),
                 "RQ1: Rambachan and Roth (2023) sensitivity of the average post-period coefficient",
                 "Relative-magnitudes restriction: each post-period violation of parallel trends is "
                 "at most M̄ times the largest pre-period one. Robust sets from the HonestDiD R "
                 "package (step 08); the breakdown M̄ is the largest M̄ at which zero is excluded.",
                 raw=raw)

    rq4u = read_result(root, MAIN, EVENT_STUDIES["rq4_uncond"][0]) if MAIN in specs else None
    rq4c = read_result(root, MAIN, EVENT_STUDIES["rq4_cond"][0]) if MAIN in specs else None
    if rq4u is not None and rq4c is not None:
        dec = read_result(root, MAIN, "rq4_decomposition.csv")
        if dec is not None and "quarter_uncond" in dec:
            dec = dec.rename(columns={"quarter_uncond": "quarter"})
        save_fig(fig_rq4(rq4u, rq4c, dec), outdir, "fig_rq4_earnings")
        if dec is not None:
            f = pd.DataFrame({
                "Quarter": dec["quarter"],
                "Without task controls": [f"{a:.3f}{stars(p)}" for a, p in
                                          zip(dec["estimate_uncond"], dec["pvalue_uncond"])],
                "With task controls": [f"{a:.3f}{stars(p)}" for a, p in
                                       zip(dec["estimate_cond"], dec["pvalue_cond"])],
                "Difference (composition)": dec["composition_component"].map("{:.3f}".format)})
            tw.write("table_rq4_decomposition", f,
                     "RQ4: earnings with and without task controls, main specification",
                     "Difference = without minus with task controls; point estimate only. "
                     + STAR_NOTE.split(". ")[1], raw=dec)

    if pre_rows:
        raw = pd.DataFrame(pre_rows)
        tw.write("table_pretrends", raw.fillna(""),
                 "Pre-period coefficients by outcome and specification",
                 "Counts of individually significant pre-period coefficients, the largest "
                 "|estimate/SE|, and the p-value of the joint Wald test that all pre-period "
                 "coefficients are zero (clustered covariance; blank for runs made before "
                 "step 07 saved it).",
                 raw=raw)

    # RQ3 hedonic
    frames = {lab[s]: d for s in specs if (d := read_result(root, s, "rq3_hedonic.csv")) is not None}
    # Eq. 7 has no exposure term, so a specification that only swaps the
    # exposure measure reproduces the main estimates; show those once.
    same = [k for k, d in frames.items() if k != lab.get(MAIN) and lab.get(MAIN) in frames
            and d[["term", "estimate", "se"]].equals(frames[lab[MAIN]][["term", "estimate", "se"]])]
    frames = {k: d for k, d in frames.items() if k not in same}
    same_note = (f" Not shown because identical to the main specification (Eq. 7 does not "
                 f"use exposure): {', '.join(same)}." if same else "")
    if frames:
        save_fig(fig_rq3(frames), outdir, "fig_rq3_task_prices")
        blocks = {k: d.set_index("term") for k, d in frames.items()}
        keys = [t for t in RQ3_TERMS if any(t in b.index for b in blocks.values())]
        f = stacked_coef_table(blocks, keys, RQ3_TERMS)
        f.columns = ["Term"] + list(f.columns[1:])
        raw = pd.concat([d.assign(specification=k) for k, d in frames.items()], ignore_index=True)
        tw.write("table_rq3_hedonic", f, "RQ3: hedonic implicit task prices (Eq. 7), outcome ln W",
                 "State-by-quarter and industry fixed effects; controls sex, age, education. "
                 + STAR_NOTE + same_note, raw=raw)

    tw.workbook(outdir / "chapter4_tables.xlsx")
    write_sources(root, outdir, specs, lab)
    return 0


def write_sources(root: pathlib.Path, outdir: pathlib.Path, specs: list, lab: dict):
    try:
        code = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=pathlib.Path(__file__).parent).stdout.strip()
    except OSError:
        code = ""
    code = os.environ.get("FIGURES_CODE_SHA", code) or "unknown"
    lines = [f"built_from: {root.resolve()}", f"figure_code_commit: {code}",
             "specifications:"] + [f"  {s}: {lab[s]}" for s in specs]
    prov = root / "logs" / "provenance.txt"
    if prov.exists():
        lines += ["", "results provenance (logs/provenance.txt):", prov.read_text()]
    (outdir / "SOURCES.txt").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".",
                    help="folder holding data/out/results*/, data/interim/ and logs/ "
                         "(repo root, or a runs/<run>/ folder from the results branch)")
    ap.add_argument("--out", default="data/out/chapter4")
    ap.add_argument("--specs", default="",
                    help="comma-separated specification folders to include, in order "
                         "(default: all results* folders found)")
    ap.add_argument("--label", action="append", default=[], metavar="FOLDER=LABEL",
                    help="readable name for a specification folder; repeatable")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    labels = dict(SPEC_LABELS)
    for item in args.label:
        k, _, v = item.partition("=")
        labels[k.strip()] = v.strip()
    specs = discover_specs(root, [s for s in args.specs.split(",") if s] or None)
    if not specs:
        print(f"[06] no result folders under {root / 'data' / 'out'}; run step 07 first")
        return 1
    return build(root, pathlib.Path(args.out), specs, labels)


if __name__ == "__main__":
    raise SystemExit(main())
