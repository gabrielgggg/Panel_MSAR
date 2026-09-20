"""1995+: common growth tau_t, RE a_i, MS-AR z only. rho<=0.99. No catch-up."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from common_trend import common_growth_trend
from panel_msar import PanelMSAR
from msar_report import compile_tex, save_cycle_pdf, _tabular, _trend_note
from run_msar_re import load_panel
from run_msar_rw import plot_tau_spaghetti, subtract_tau

YEAR_MIN = 1995
RHO_MAX = 0.99
OUT_PDF = HERE / "msar_rw_1995.pdf"
TEX = HERE / "msar_rw_1995.tex"
FIGS = HERE / "figs"
SAMPLE_CSV = HERE / "rw1995_sample.csv"
TAU_CSV = HERE / "rw1995_tau.csv"

SPEC = dict(
    key="rw1995_re",
    title="RE intercepts around the common RW trend (no catch-up)",
)


def _tex_preamble(tau_fig, tau_info):
    return [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,graphicx,setspace,caption,hyperref}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathpazo}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\captionsetup{font=small,skip=6pt}",
        r"\title{Panel MS-AR(1) from 1995: common RW trend and RE intercepts\\[0.4em]"
        r"\large Seasonally adjusted real GDP per worker}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Model}",
        r"Sample starts in 1995Q1. Two-step. Step 1 is a common trend from average "
        r"growth of countries observed in consecutive quarters:",
        r"\begin{align}",
        r"\Delta\tau_t &= \frac{1}{n_t^{\mathrm{cont}}}\sum_{i\in C_t\cap C_{t-1}} (y_{it}-y_{i,t-1}), \\",
        r"\tau_t &= \tau_{t_0}+\sum_{s=t_0+1}^{t}\Delta\tau_s.",
        r"\end{align}",
        r"The level of $\tau$ is pinned to the balanced-core mean at $t_0$. "
        rf"Mean $\Delta\tau$ is {tau_info['mu']:.4f} per quarter "
        rf"({tau_info.get('n_core', 0)} countries present in every quarter). "
        r"Step 2 is estimated on $y_{it}^\ast=y_{it}-\hat\tau_t$:",
        r"\begin{align}",
        r"y_{it}^\ast &= a_i + z_{it}, \\",
        r"a_i &\sim \mathcal{N}(\alpha,\omega^2), \\",
        r"z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it}) + \rho(s_{it})\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}.",
        r"\end{align}",
        r"No catch-up term. No extra $gt$. Three regimes; median $\mu$ pinned at 0; "
        rf"$\sigma$ and $\rho$ switch; $|\rho|<{RHO_MAX}$. "
        r"Random intercepts: 11-point Gauss--Hermite. "
        r"Plotted cycles use posterior-mean $a_i$. "
        r"$\tau_t$ is discarded for the structural MS-AR. "
        r"Standard errors ignore step-1 uncertainty.",
        r"\begin{figure}[h]",
        r"\centering",
        rf"\includegraphics[width=0.88\textwidth]{{{tau_fig.as_posix()}}}",
        r"\caption{Faint lines: country log real GDP per worker. "
        r"Thick line: common $\tau_t$ from average growth of continuing countries.}",
        r"\end{figure}",
    ]


def main():
    raw = load_panel()
    raw = raw.loc[raw["time"] >= YEAR_MIN].copy()
    raw = raw.sort_values(["country", "time"], kind="mergesort").reset_index(drop=True)
    print(
        f"Sample from {YEAR_MIN}: {raw.country.nunique()} countries, "
        f"{len(raw)} obs, {raw.period.min()}--{raw.period.max()}.",
        flush=True,
    )
    tau_info = common_growth_trend(
        raw["country"].to_numpy(), raw["time"].to_numpy(), raw["y"].to_numpy()
    )
    grid = tau_info["times"]
    print(
        f"Common growth trend: {grid.size} dates, n per date "
        f"{int(tau_info['n'].min())}--{int(tau_info['n'].max())}, "
        f"core {tau_info['n_core']}, mean dtau={tau_info['mu']:.5f}",
        flush=True,
    )
    df = subtract_tau(raw, grid, tau_info["tau"])
    df.to_csv(SAMPLE_CSV, index=False)
    pd.DataFrame({
        "time": grid, "ybar": tau_info["ybar"], "tau": tau_info["tau"],
        "n": tau_info["n"], "n_delta": tau_info["n_delta"],
    }).to_csv(TAU_CSV, index=False)
    FIGS.mkdir(exist_ok=True)
    tau_fig = FIGS / "tau_rw_1995.pdf"
    plot_tau_spaghetti(df, grid, tau_info["tau"], tau_fig)
    sample_line = (
        rf"2026-09-17 SA panel, $y_{{it}}=a_i+\tau_t+z_{{it}}$, from {YEAR_MIN}. "
        rf"{df.country.nunique()} countries, {len(df)} observations, "
        rf"{df.period.min()}--{df.period.max()}. $|\rho|<{RHO_MAX}$."
    )
    print(sample_line, flush=True)
    spec = SPEC
    print("\n" + "=" * 72, flush=True)
    print(spec["title"], flush=True)
    mod = PanelMSAR(
        n_regimes=3,
        common_rho=False,
        common_sigma=False,
        random_intercepts=True,
        convergence=False,
        include_trend=False,
        zero_mu=False,
        min_t=12,
        rho_max=RHO_MAX,
    )
    res = mod.fit(
        df["country"], df["time"], df["y"],
        n_starts=4, maxiter=400, seed=1,
        compute_se=True, store_filtered=True, verbose=True,
    )
    print(res, flush=True)
    fig = FIGS / f"cycle_{spec['key']}.pdf"
    save_cycle_pdf(res, fig, spec["title"])
    rel = fig.relative_to(HERE)
    tau_rel = tau_fig.relative_to(HERE)
    ez = res.params.get("Ez")
    se = res.se_params or {}
    se_ez = se.get("Ez")
    ez_line = ""
    if ez is not None and np.isfinite(ez):
        extra = ""
        if se_ez is not None and np.isfinite(float(se_ez)):
            extra = f" ({float(se_ez):.4f})"
        ez_line = rf" Ergodic mean of the cycle $E[z]={float(ez):.4f}${extra}."
    parts = _tex_preamble(tau_rel, tau_info)
    parts += [
        r"\clearpage",
        rf"\section{{{spec['title']}}}",
        sample_line + rf" Fitted countries: {res.n_countries}. "
        rf"Observations: {res.nobs}. Log-likelihood: {res.loglik:.2f}."
        + ez_line,
        _trend_note(res),
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Regime AR(1), transition matrix, and ergodic $\pi$. Columns are regimes.}",
        _tabular(res),
        r"\end{table}",
        r"\begin{figure}[h]",
        r"\centering",
        rf"\includegraphics[width=0.75\textwidth]{{{rel.as_posix()}}}",
        r"\caption{Country cycles after removing $\tau_t$ and posterior-mean $a_i$.}",
        r"\end{figure}",
        r"\end{document}",
    ]
    TEX.write_text("\n".join(parts) + "\n", encoding="utf-8")
    compile_tex(TEX)
    print(f"\nWrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
