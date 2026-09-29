"""1985 through the end of the sample, country fixed effects, drop US/Germany/Japan/China.

Two-step: y = a_i + g t + z. g is continuing-country growth, a_i zeros
the mean of z inside each country, and z is an MS-AR(1) with E[z]=0
and |rho|<0.99. The window includes the COVID years and runs through
the last quarter in the vintage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from panel_msar import PanelMSAR
from msar_report import compile_tex, save_cycle_pdf, _tabular, _trend_note
from load_panel import load_panel

YEAR_MIN = 1985.0
RHO_MAX = 0.99
DROP = {"USA", "DEU", "JPN", "CHN"}
OUT_PDF = HERE / "msar_fe_1985.pdf"
TEX = HERE / "msar_fe_1985.tex"
FIGS = HERE / "figs"
SAMPLE_CSV = HERE / "fe_1985_sample.csv"
SPEC_TITLE = (
    "Country intercepts and common growth, 1985 through the end of the sample, "
    "ex US, Germany, Japan, China"
)


def _tex(sample_line, res, fig, absent, span):
    drop_note = r"United States, Germany, and Japan are excluded."
    if absent:
        drop_note += " China is not in this vintage."
    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,graphicx,setspace,caption,hyperref}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathpazo}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\captionsetup{font=small,skip=6pt}",
        r"\title{Panel MS-AR(1): country intercepts and common growth\\[0.4em]"
        r"\large 1985 through the end of the sample, excluding the US, Germany, Japan, and China}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Model}",
        r"Seasonally adjusted real GDP per worker, 2026-09-17 vintage, "
        + span
        + r". "
        + drop_note,
        r"\begin{align}",
        r"y_{it} &= a_i + g\, t + z_{it}, \\",
        r"s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}), \\",
        r"z_{i,t+1} &= \bigl(1-\rho(s_{i,t+1})\bigr)\mu(s_{i,t+1}) + \rho(s_{i,t+1})\, z_{it} + \sigma(s_{i,t+1})\,\varepsilon_{it}.",
        r"\end{align}",
        r"The regime $s_{it}$ is the one that produced $z_{it}$. "
        r"$\varepsilon_{it}$ is the innovation. "
        r"For each pair of consecutive dates, $g$ uses the average growth "
        r"of countries observed on both dates, and those date-pair averages "
        r"are then averaged with equal weight. "
        r"$a_i$ is the mean of $y_{it}-gt$ on country $i$, so "
        r"$z_{it}=y_{it}-a_i-gt$ has mean zero inside each country. "
        r"Three regimes. The MS-AR imposes $E[z]=0$. "
        rf"$\sigma$ and $\rho$ switch, with $|\rho|<{RHO_MAX}$. "
        r"Standard errors are delta-method from a numerical Hessian of the "
        r"cycle parameters. $a_i$ and $g$ have none.",
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
        r"\caption{Country cycles $z_{it}=y_{it}-a_i-gt$.}",
        r"\end{figure}",
        r"\end{document}",
    ]
    return "\n".join(parts) + "\n"


def main():
    df = load_panel()
    absent = sorted(DROP - set(df["country"].unique()))
    present_drop = sorted(DROP & set(df["country"].unique()))
    sub = df.loc[
        (df["time"] >= YEAR_MIN)
        & ~df["country"].isin(DROP)
    ].copy()
    sub.to_csv(SAMPLE_CSV, index=False)
    print(f"Dropped (in panel): {', '.join(present_drop)}", flush=True)
    if absent:
        print(f"Requested drop but not in panel: {', '.join(absent)}", flush=True)
    span = rf"{sub.period.min()} through {sub.period.max()}"
    sample_line = (
        rf"2026-09-17 SA panel, {span}, excluding the US, Germany, and Japan. "
        rf"$y_{{it}}=a_i+gt+z_{{it}}$, $g$ continuing-country growth, $E[z]=0$. "
        rf"{sub.country.nunique()} countries, {len(sub)} observations, "
        rf"{sub.period.min()}--{sub.period.max()}."
    )
    print(sample_line, flush=True)
    FIGS.mkdir(exist_ok=True)
    mod = PanelMSAR(
        n_regimes=3,
        common_rho=False,
        min_t=12,
        rho_max=RHO_MAX,
        country_fe=True,
    )
    res = mod.fit(
        sub["country"], sub["time"], sub["y"],
        n_starts=9, maxiter=400, seed=1,
        compute_se=True, store_filtered=True, verbose=True,
    )
    print(res, flush=True)
    print(f"check E[z]={float(res.params.get('Ez', np.nan)):.6e}", flush=True)
    fig = FIGS / "cycle_fe_1985.pdf"
    save_cycle_pdf(res, fig, SPEC_TITLE)
    TEX.write_text(
        _tex(sample_line, res, fig.relative_to(HERE), absent, span),
        encoding="utf-8",
    )
    compile_tex(TEX)
    print(f"\nWrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
