"""Full sample: common a + g t, 3 regimes, E[z]=0, |rho|<0.99."""
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
from run_msar_re import load_panel

RHO_MAX = 0.99
OUT_PDF = HERE / "msar_common_full.pdf"
TEX = HERE / "msar_common_full.tex"
FIGS = HERE / "figs"
SAMPLE_CSV = HERE / "common_full_sample.csv"
SPEC_TITLE = "Common intercept and linear trend, full sample"


def _tex(sample_line, res, fig):
    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,graphicx,setspace,caption,hyperref}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathpazo}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\captionsetup{font=small,skip=6pt}",
        r"\title{Panel MS-AR(1): common intercept and linear trend\\[0.4em]"
        r"\large Full sample, seasonally adjusted real GDP per worker}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Model}",
        r"All countries and all years in the 2026-09-17 panel. "
        r"No country random effects and no catch-up term.",
        r"\begin{align}",
        r"y_{it} &= a + g\, t + z_{it}, \\",
        r"z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it}) + \rho(s_{it})\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}, \\",
        r"s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}).",
        r"\end{align}",
        r"Three regimes. The unconditional mean of the cycle is restricted to "
        r"$E[z]=0$ (the median regime mean is not pinned at 0). "
        rf"$\sigma$ and $\rho$ switch, with $|\rho|<{RHO_MAX}$. "
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
        r"\caption{Country cycles after removing the common intercept and the linear trend.}",
        r"\end{figure}",
        r"\end{document}",
    ]
    return "\n".join(parts) + "\n"


def main():
    df = load_panel()
    sub = df.copy()
    sub.to_csv(SAMPLE_CSV, index=False)
    sample_line = (
        rf"2026-09-17 SA panel, full sample, $y_{{it}}=a+gt+z_{{it}}$, $E[z]=0$. "
        rf"{sub.country.nunique()} countries, {len(sub)} observations, "
        rf"{sub.period.min()}--{sub.period.max()}."
    )
    print(sample_line, flush=True)
    FIGS.mkdir(exist_ok=True)
    mod = PanelMSAR(
        n_regimes=3,
        common_rho=False,
        common_sigma=False,
        random_intercepts=False,
        convergence=False,
        include_trend=True,
        zero_ez=True,
        min_t=12,
        rho_max=RHO_MAX,
    )
    res = mod.fit(
        sub["country"], sub["time"], sub["y"],
        n_starts=9, maxiter=400, seed=1,
        compute_se=True, store_filtered=True, verbose=True,
    )
    print(res, flush=True)
    print(f"check E[z]={float(res.params.get('Ez', np.nan)):.6e}", flush=True)
    fig = FIGS / "cycle_common_full.pdf"
    save_cycle_pdf(res, fig, SPEC_TITLE)
    TEX.write_text(_tex(sample_line, res, fig.relative_to(HERE)), encoding="utf-8")
    compile_tex(TEX)
    print(f"\nWrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
