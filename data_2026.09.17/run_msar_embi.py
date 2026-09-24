"""EMBI-ever countries: RE intercepts, common linear trend, 3-regime MS-AR."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from panel_msar import PanelMSAR
from msar_report import compile_tex, save_cycle_pdf, _tabular, _trend_note
from run_msar_re import YCOL, load_panel

RHO_MAX = 0.99
YEAR_MIN = 1990
N_DROP_POOR = 3
OUT_PDF = HERE / "msar_embi_1990.pdf"
TEX = HERE / "msar_embi_1990.tex"
FIGS = HERE / "figs"
SAMPLE_CSV = HERE / "embi_1990_sample.csv"

# In J.P. Morgan EMBI Global as of April 2025 (Brookings Hutchins WP104,
# Tables 1-3) or named in an earlier published EMBI Global coverage list
# (Korea, Thailand, Croatia, Tunisia). ISO3 as in this panel (ROM = Romania).
EMBI_EVER = {
    "ARG", "AZE", "BOL", "BRA", "BGR", "CMR", "CHL", "COL", "CRI", "DOM",
    "ECU", "EGY", "SLV", "GEO", "GHA", "GTM", "HND", "HRV", "HUN", "IND",
    "IDN", "JOR", "KAZ", "KEN", "KOR", "LKA", "MYS", "MAR", "MEX", "NGA",
    "PRY", "PER", "PHL", "POL", "ROM", "SEN", "SRB", "ZAF", "THA", "TUN",
    "TUR", "URY", "ZMB",
}

SPEC_TITLE = "EMBI countries from 1990, three poorest in 2019 dropped"


def _tex(sample_line, res, fig, n_kept, missing, dropped):
    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,graphicx,setspace,caption,hyperref}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathpazo}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\captionsetup{font=small,skip=6pt}",
        r"\title{Panel MS-AR(1) for EMBI countries from 1990\\[0.4em]"
        r"\large Seasonally adjusted real GDP per worker}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Model}",
        r"Countries that are in the J.P.\ Morgan EMBI Global as of April 2025, "
        r"or that appear on an earlier published EMBI Global coverage list "
        r"(Korea, Thailand, Croatia, and Tunisia). "
        rf"Of that list, {n_kept} countries are in the 2026-09-17 panel"
        + (rf" (not in the panel: {missing}). " if missing else ". ")
        + rf"The three with the lowest 2019 real GDP per worker are dropped ({dropped}). "
        + rf"Estimation starts in {YEAR_MIN}.",
        r"One specification. Joint panel Markov-switching AR(1) around a common "
        r"linear trend, with Normal random intercepts. No catch-up term.",
        r"\begin{align}",
        r"y_{it} &= a_i + g\, t + z_{it}, \\",
        r"a_i &\sim \mathcal{N}(\alpha,\omega^2), \\",
        r"z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it}) + \rho(s_{it})\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}, \\",
        r"s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}).",
        r"\end{align}",
        r"Outcome is log seasonally adjusted real GDP per worker. "
        r"Three regimes. The unconditional mean of the cycle is restricted to $E[z]=0$ "
        r"(the median regime mean is not pinned at 0). $\sigma$ and $\rho$ switch. "
        rf"$|\rho|<{RHO_MAX}$. "
        r"Random intercepts use 11-point Gauss--Hermite. "
        r"Plotted cycles use the posterior mean of $a_i$. "
        r"Standard errors are delta-method from a numerical Hessian.",
        r"\clearpage",
        rf"\section{{{SPEC_TITLE}}}",
        sample_line
        + rf" Fitted countries: {res.n_countries}. "
        + rf"Observations: {res.nobs}. Log-likelihood: {res.loglik:.2f}.",
        _trend_note(res),
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Regime AR(1), transition matrix, and ergodic $\pi$. Columns are regimes.}",
        _tabular(res),
        r"\end{table}",
        r"\begin{figure}[h]",
        r"\centering",
        rf"\includegraphics[width=0.75\textwidth]{{{fig.as_posix()}}}",
        r"\caption{Country cycles after removing the posterior-mean intercept and the common linear trend.}",
        r"\end{figure}",
        r"\end{document}",
    ]
    return "\n".join(parts) + "\n"


def poorest_in_2019(df, countries, n_drop):
    """Lowest mean real GDP per worker in calendar year 2019."""
    y2019 = df.loc[(df["country"].isin(countries)) & (df["year"] == 2019)]
    if y2019.empty:
        raise ValueError("No 2019 observations for the EMBI subsample.")
    means = y2019.groupby("country")[YCOL].mean().sort_values()
    dropped = list(means.index[:n_drop])
    print("2019 mean real GDP per worker, poorest first:", flush=True)
    for c, v in means.items():
        mark = "  DROP" if c in dropped else ""
        print(f"  {c}  {v:,.0f}{mark}", flush=True)
    return dropped


def main():
    df = load_panel()
    have = set(df["country"].unique())
    keep = sorted(EMBI_EVER & have)
    missing = ", ".join(sorted(EMBI_EVER - have))
    dropped = poorest_in_2019(df, keep, N_DROP_POOR)
    keep = [c for c in keep if c not in dropped]
    sub = df.loc[df["country"].isin(keep) & (df["time"] >= YEAR_MIN)].copy()
    sub.to_csv(SAMPLE_CSV, index=False)
    print(f"EMBI countries kept ({len(keep)}): {', '.join(keep)}", flush=True)
    print(f"Dropped as poorest in 2019: {', '.join(dropped)}", flush=True)
    if missing:
        print(f"On the EMBI list but not in the panel: {missing}", flush=True)
    sample_line = (
        rf"EMBI subsample of the 2026-09-17 SA panel from {YEAR_MIN}, "
        rf"dropping {', '.join(dropped)} (lowest 2019 real GDP per worker). "
        rf"$y_{{it}}=a_i+gt+z_{{it}}$, $E[z]=0$. "
        rf"{sub.country.nunique()} countries, {len(sub)} observations, "
        rf"{sub.period.min()}--{sub.period.max()}."
    )
    print(sample_line, flush=True)
    FIGS.mkdir(exist_ok=True)
    mod = PanelMSAR(
        n_regimes=3,
        common_rho=False,
        common_sigma=False,
        random_intercepts=True,
        include_trend=True,
        zero_ez=True,
        min_t=12,
        rho_max=RHO_MAX,
    )
    res = mod.fit(
        sub["country"], sub["time"], sub["y"],
        n_starts=4, maxiter=400, seed=1,
        compute_se=True, store_filtered=True, verbose=True,
    )
    print(res, flush=True)
    fig = FIGS / "cycle_embi_1990.pdf"
    save_cycle_pdf(res, fig, SPEC_TITLE)
    TEX.write_text(
        _tex(
            sample_line, res, fig.relative_to(HERE),
            len(keep) + len(dropped), missing, ", ".join(dropped),
        ),
        encoding="utf-8",
    )
    compile_tex(TEX)
    print(f"\nWrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
