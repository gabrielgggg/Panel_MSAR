"""Full sample: common growth tau_t, RE a_i, no catch-up, E[z]=0, 3 regimes."""
from __future__ import annotations

import json
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

RHO_MAX = 0.99
OUT_PDF = HERE / "msar_full_rw.pdf"
TEX = HERE / "msar_full_rw.tex"
FIGS = HERE / "figs"
SAMPLE_CSV = HERE / "full_rw_sample.csv"
TAU_CSV = HERE / "full_rw_tau.csv"
CYCLE_JSON = HERE / "full_rw_cycle.json"
SPEC_TITLE = "Full sample: RE intercepts around a common RW trend"


def _tex(sample_line, res, cyc, tau_fig, tau_info):
    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,graphicx,setspace,caption,hyperref}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathpazo}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\captionsetup{font=small,skip=6pt}",
        r"\title{Panel MS-AR(1), full sample\\[0.4em]"
        r"\large Common random-walk trend and random intercepts}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Model}",
        r"All countries and all years in the 2026-09-17 seasonally adjusted "
        r"real GDP per worker panel. No catch-up term.",
        r"Step 1 builds a common trend from average growth of countries observed "
        r"in consecutive quarters:",
        r"\begin{align}",
        r"\Delta\tau_t &= \frac{1}{n_t^{\mathrm{cont}}}\sum_{i\in C_t\cap C_{t-1}}(y_{it}-y_{i,t-1}), \\",
        r"\tau_t &= \tau_{t_0}+\sum_{s=t_0+1}^{t}\Delta\tau_s.",
        r"\end{align}",
        rf"Mean $\Delta\tau$ is {tau_info['mu']:.4f} per quarter "
        rf"({tau_info.get('n_core', 0)} countries present every quarter). "
        r"A common shift in $\tau$ is absorbed by $\alpha$. "
        r"Step 2 uses $y_{it}^\ast=y_{it}-\hat\tau_t$:",
        r"\begin{align}",
        r"y_{it}^\ast &= a_i + z_{it}, \\",
        r"a_i &\sim \mathcal{N}(\alpha,\omega^2), \\",
        r"z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it}) + \rho(s_{it})\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}.",
        r"\end{align}",
        r"Three regimes. $E[z]=0$. "
        rf"$\sigma$ and $\rho$ switch, $|\rho|<{RHO_MAX}$. "
        r"Random intercepts use 11-point Gauss--Hermite. "
        r"Standard errors ignore step-1 uncertainty.",
        r"\begin{figure}[h]",
        r"\centering",
        rf"\includegraphics[width=0.88\textwidth]{{{tau_fig.as_posix()}}}",
        r"\caption{Faint lines: country log real GDP per worker. Thick line: common $\tau_t$.}",
        r"\end{figure}",
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
        rf"\includegraphics[width=0.75\textwidth]{{{cyc.as_posix()}}}",
        r"\caption{Country cycles after removing $\tau_t$ and the posterior-mean intercept.}",
        r"\end{figure}",
        r"\end{document}",
    ]
    return "\n".join(parts) + "\n"


def main():
    df = load_panel()
    df = df.sort_values(["country", "time"], kind="mergesort").reset_index(drop=True)
    print(
        f"Full sample: {df.country.nunique()} countries, {len(df)} obs, "
        f"{df.period.min()}--{df.period.max()}.",
        flush=True,
    )
    tau_info = common_growth_trend(
        df["country"].to_numpy(), df["time"].to_numpy(), df["y"].to_numpy()
    )
    grid = tau_info["times"]
    print(
        f"tau: {grid.size} dates, n {int(tau_info['n'].min())}--{int(tau_info['n'].max())}, "
        f"core {tau_info['n_core']}, mean dtau={tau_info['mu']:.5f}",
        flush=True,
    )
    df = subtract_tau(df, grid, tau_info["tau"])
    df.to_csv(SAMPLE_CSV, index=False)
    pd.DataFrame({
        "time": grid, "ybar": tau_info["ybar"], "tau": tau_info["tau"],
        "n": tau_info["n"], "n_delta": tau_info["n_delta"],
    }).to_csv(TAU_CSV, index=False)
    FIGS.mkdir(exist_ok=True)
    tau_fig = FIGS / "tau_full_rw.pdf"
    plot_tau_spaghetti(df, grid, tau_info["tau"], tau_fig)
    sample_line = (
        rf"Full 2026-09-17 sample, $y_{{it}}=a_i+\tau_t+z_{{it}}$, $E[z]=0$. "
        rf"{df.country.nunique()} countries, {len(df)} observations, "
        rf"{df.period.min()}--{df.period.max()}."
    )
    print(sample_line, flush=True)
    mod = PanelMSAR(
        n_regimes=3,
        common_rho=False,
        common_sigma=False,
        random_intercepts=True,
        include_trend=False,
        zero_ez=True,
        convergence=False,
        min_t=12,
        rho_max=RHO_MAX,
    )
    res = mod.fit(
        df["country"], df["time"], df["y"],
        n_starts=4, maxiter=400, seed=1,
        compute_se=True, store_filtered=True, verbose=True,
    )
    print(res, flush=True)
    print(f"check E[z]={float(res.params.get('Ez', np.nan)):.6e}", flush=True)
    cycle = {
        "mu": [float(x) for x in np.asarray(res.params["mu"]).ravel()],
        "rho": [float(x) for x in np.asarray(res.params["rho"]).ravel()],
        "sigma": [float(x) for x in np.asarray(res.params["sigma"]).ravel()],
    }
    CYCLE_JSON.write_text(json.dumps(cycle, indent=2), encoding="utf-8")
    cyc = FIGS / "cycle_full_rw.pdf"
    save_cycle_pdf(res, cyc, SPEC_TITLE)
    TEX.write_text(
        _tex(sample_line, res, cyc.relative_to(HERE), tau_fig.relative_to(HERE), tau_info),
        encoding="utf-8",
    )
    compile_tex(TEX)
    print(f"\nWrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
