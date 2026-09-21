"""Spec 1 only: common a + g t, rho fixed at 0.99 in every regime."""
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

RHO_FIXED = 0.99
OUT_PDF = HERE / "msar_spec1_rho99.pdf"
TEX = HERE / "msar_spec1_rho99.tex"
FIGS = HERE / "figs"

SPEC = dict(
    key="spec1_rho99",
    title="Common intercept and linear trend, rho fixed at 0.99",
)


def _tex(sample_line, res, fig):
    ez = res.params.get("Ez")
    se = res.se_params or {}
    se_ez = se.get("Ez")
    ez_line = ""
    if ez is not None and np.isfinite(float(ez)):
        extra = ""
        if se_ez is not None and np.isfinite(float(se_ez)):
            extra = f" ({float(se_ez):.4f})"
        ez_line = rf" Ergodic mean of the cycle $E[z]={float(ez):.4f}${extra}."
    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,graphicx,setspace,caption,hyperref}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathpazo}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\captionsetup{font=small,skip=6pt}",
        r"\title{Panel MS-AR(1): common linear trend, $\rho=0.99$\\[0.4em]"
        r"\large Seasonally adjusted real GDP per worker}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Model}",
        r"One specification. Joint panel Markov-switching AR(1) around a common "
        r"linear trend. No country-specific random effects. Outcome is log "
        r"seasonally adjusted real GDP per worker (2026-09-17 vintage), full sample.",
        r"\begin{align}",
        r"y_{it} &= a + g\, t + z_{it}, \\",
        r"z_{i,t+1} &= \bigl(1-\rho\bigr)\mu(s_{it}) + \rho\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}, \\",
        r"s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}).",
        r"\end{align}",
        rf"Three regimes. $\rho={RHO_FIXED}$ is fixed and common to every regime. "
        r"Median $\mu$ pinned at 0. $\sigma$ switches. "
        r"Standard errors are delta-method from a numerical Hessian "
        r"(not reported for the fixed $\rho$).",
        r"\clearpage",
        rf"\section{{{SPEC['title']}}}",
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
        rf"\includegraphics[width=0.75\textwidth]{{{fig.as_posix()}}}",
        r"\caption{Country cycles after removing the common intercept and linear trend.}",
        r"\end{figure}",
        r"\end{document}",
    ]
    return "\n".join(parts) + "\n"


def main():
    df = load_panel()
    sample_line = (
        rf"2026-09-17 SA panel, full sample, $y_{{it}}=a+gt+z_{{it}}$, "
        rf"$\rho={RHO_FIXED}$ fixed. "
        rf"{df.country.nunique()} countries, {len(df)} observations, "
        rf"{df.period.min()}--{df.period.max()}."
    )
    print(sample_line, flush=True)
    FIGS.mkdir(exist_ok=True)
    mod = PanelMSAR(
        n_regimes=3,
        common_rho=True,
        rho_value=RHO_FIXED,
        common_sigma=False,
        random_intercepts=False,
        convergence=False,
        include_trend=True,
        zero_mu=False,
        min_t=12,
    )
    res = mod.fit(
        df["country"], df["time"], df["y"],
        n_starts=4, maxiter=400, seed=1,
        compute_se=True, store_filtered=True, verbose=True,
    )
    print(res, flush=True)
    fig = FIGS / f"cycle_{SPEC['key']}.pdf"
    save_cycle_pdf(res, fig, SPEC["title"])
    TEX.write_text(_tex(sample_line, res, fig.relative_to(HERE)), encoding="utf-8")
    compile_tex(TEX)
    print(f"\nWrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
