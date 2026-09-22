"""RE intercepts, common linear trend, E[z]=0 instead of median mu=0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from panel_msar import PanelMSAR
from msar_report import compile_tex, save_cycle_pdf, _tabular, _trend_note
from run_msar_re import load_panel

RHO_MAX = 0.99
OUT_PDF = HERE / "msar_re_ez0.pdf"
TEX = HERE / "msar_re_ez0.tex"
FIGS = HERE / "figs"
SPEC_TITLE = "Random intercepts, common linear trend, E[z]=0"


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
        r"\title{Panel MS-AR(1): random intercepts with $E[z]=0$\\[0.4em]"
        r"\large Seasonally adjusted real GDP per worker}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Model}",
        r"One specification. Joint panel Markov-switching AR(1) around a common "
        r"linear trend, with Normal random intercepts. Outcome is log seasonally "
        r"adjusted real GDP per worker (2026-09-17 vintage), full sample.",
        r"\begin{align}",
        r"y_{it} &= a_i + g\, t + z_{it}, \\",
        r"a_i &\sim \mathcal{N}(\alpha,\omega^2), \\",
        r"z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it}) + \rho(s_{it})\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}, \\",
        r"s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}).",
        r"\end{align}",
        r"Three regimes. $\sigma$ and $\rho$ switch. "
        rf"$|\rho|<{RHO_MAX}$. "
        r"The intercept and the regime means are not separately identified, so "
        r"the unconditional mean of the cycle is restricted to $E[z]=0$ "
        r"(the median regime mean is \emph{not} pinned at 0). "
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


def main():
    df = load_panel()
    sample_line = (
        rf"2026-09-17 SA panel, full sample, $y_{{it}}=a_i+gt+z_{{it}}$, $E[z]=0$. "
        rf"{df.country.nunique()} countries, {len(df)} observations, "
        rf"{df.period.min()}--{df.period.max()}."
    )
    print(sample_line, flush=True)
    FIGS.mkdir(exist_ok=True)
    mod = PanelMSAR(
        n_regimes=3,
        common_rho=False,
        common_sigma=False,
        random_intercepts=True,
        zero_ez=True,
        include_trend=True,
        min_t=12,
        rho_max=RHO_MAX,
    )
    res = mod.fit(
        df["country"], df["time"], df["y"],
        n_starts=4, maxiter=400, seed=1,
        compute_se=True, store_filtered=True, verbose=True,
    )
    print(res, flush=True)
    ez = float(res.params.get("Ez", np.nan))
    print(f"check E[z]={ez:.6e}", flush=True)
    fig = FIGS / "cycle_re_ez0.pdf"
    save_cycle_pdf(res, fig, SPEC_TITLE)
    TEX.write_text(_tex(sample_line, res, fig.relative_to(HERE)), encoding="utf-8")
    compile_tex(TEX)
    print(f"\nWrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
