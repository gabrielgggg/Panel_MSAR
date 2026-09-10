"""Two linear-trend specs on SOE data from 1970Q1; LaTeX report with SEs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from panel_msar import PanelMSAR
from prepare_soe_panel import load_panel
from run_soe_spec_report import (
    compile_tex,
    save_cycle_pdf,
    _tabular,
    _trend_note,
)

OUT_PDF = HERE / "soe_msar_linear_1970.pdf"
TEX = HERE / "soe_msar_linear_1970.tex"
FIGS = HERE / "figs"

SPECS = [
    dict(
        key="lin1970_common_ag",
        title="Common intercept and linear trend",
        country_intercepts=False,
        country_trends=False,
    ),
    dict(
        key="lin1970_ai_common_g",
        title="Country-specific intercepts, common linear trend",
        country_intercepts=True,
        country_trends=False,
    ),
]


def _tex_report(sample_line, fitted):
    parts = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,graphicx,setspace,caption,hyperref}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathpazo}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\captionsetup{font=small,skip=6pt}",
        r"\title{Panel MS-AR(1) on SOE log real GDP per worker, 1970Q1 onward}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Model}",
        r"Joint panel Markov-switching AR(1) around a \emph{linear} trend, "
        r"with three regimes:",
        r"\begin{align}",
        r"y_{it} &= a_i + g\, t + z_{it}, \\",
        r"z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it}) + \rho(s_{it})\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}, \\",
        r"s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}).",
        r"\end{align}",
        r"The trend is linear in calendar time $t$ (year-fraction; $g$ is per year). "
        r"The intercept $a_i$ is either common across countries or country-specific; "
        r"the slope $g$ is always common. The cycle $z_{it}$ is a three-regime AR(1). "
        r"Countries are independent given shared Markov parameters; latent paths "
        r"$s_{it}$ are country-specific. The regime dated $t$ governs the transition "
        r"from $z_t$ to $z_{t+1}$. $\sigma$ and $\rho$ switch with the regime. "
        r"The median $\mu$ is pinned at 0 after ordering. "
        r"$|\rho|<0.995$. Likelihood: Hamilton filter per country, summed.",
        r"\subsection*{Specifications}",
        r"\begin{enumerate}",
        r"\item Common intercept $a$ and common linear slope $g$ (joint MLE).",
        r"\item Country intercepts $a_i$ and common $g$ ($a_i$ profiled).",
        r"\end{enumerate}",
        r"Standard errors (in parentheses) are delta-method from a numerical Hessian "
        r"on shared parameters, or country-cluster bootstrap when the Hessian is "
        r"not invertible. A dashed entry is a pinned coefficient. "
        r"$\pi$ is the ergodic distribution of $\Pi$. $E[z]$ is the long-run mean "
        r"of the cycle implied by $\mu$, $\rho$, and $\Pi$.",
    ]
    for spec, res, fig in fitted:
        se_note = ""
        if res.se_params is None:
            se_note = " Standard errors omitted."
        elif res.warnings and any("bootstrap" in w.lower() for w in res.warnings):
            se_note = " Standard errors from country bootstrap."
        ez = res.params.get("Ez")
        se = res.se_params or {}
        se_ez = se.get("Ez")
        ez_line = ""
        if ez is not None and np.isfinite(ez):
            extra = ""
            if se_ez is not None and np.isfinite(float(se_ez)):
                extra = f" ({float(se_ez):.4f})"
            ez_line = rf" Ergodic mean of the cycle $E[z]={float(ez):.4f}${extra}."
        parts += [
            r"\clearpage",
            rf"\section{{{spec['title']}}}",
            sample_line + rf" Fitted countries: {res.n_countries}. "
            rf"Observations: {res.nobs}. Log-likelihood: {res.loglik:.2f}."
            + se_note + ez_line,
            _trend_note(res),
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Regime AR(1), transition matrix, and ergodic $\pi$. Columns are regimes.}",
            _tabular(res),
            r"\end{table}",
            r"\begin{figure}[h]",
            r"\centering",
            rf"\includegraphics[width=0.75\textwidth]{{{fig.as_posix()}}}",
            r"\caption{Country cycles after removing the linear trend.}",
            r"\end{figure}",
        ]
    parts.append(r"\end{document}")
    return "\n".join(parts) + "\n"


def fit_spec(df, spec, verbose=True):
    print("\n" + "=" * 72, flush=True)
    print(spec["title"], flush=True)
    print("=" * 72, flush=True)
    mod = PanelMSAR(
        n_regimes=3,
        common_rho=False,
        common_sigma=False,
        country_intercepts=spec["country_intercepts"],
        country_trends=False,
        two_step=False,
        zero_mu=False,
        min_t=12,
    )
    use_boot = bool(spec["country_intercepts"])
    res = mod.fit(
        df["country"],
        df["time"],
        df["y"],
        n_starts=8,
        maxiter=400,
        seed=1,
        compute_se=not use_boot,
        store_filtered=True,
        verbose=verbose,
    )
    if use_boot:
        print("Country bootstrap SEs (B=40, warm start)...", flush=True)
        se_b, n_ok = mod.bootstrap_se(
            df["country"], df["time"], df["y"],
            res.theta, B=40, seed=11, maxiter=180, verbose=True,
        )
        if se_b is not None:
            res.se_params = se_b
            res.warnings.append(
                f"Country bootstrap SEs from {n_ok} successful draws (B=40)."
            )
            print(f"  bootstrap draws kept: {n_ok}/40", flush=True)
        else:
            res.warnings.append(
                f"Bootstrap produced only {n_ok} usable draws; SEs omitted."
            )
    txt = HERE / f"output_{spec['key']}_SOE.txt"
    header = (
        f"SOE log real GDP per worker from 1970Q1\n{spec['title']}\n"
        f"n_regimes=3  country_intercepts={spec['country_intercepts']}  "
        f"country_trends=False  common_rho=False  rho_max=0.995  "
        f"compute_se=True\n\n"
    )
    txt.write_text(header + res.summary() + "\n", encoding="utf-8")
    print(res, flush=True)
    print(f"Wrote {txt}", flush=True)
    return res


def main():
    df = load_panel()
    df = df.loc[df["time"] >= 1970.0].copy()
    sample_line = (
        rf"SOE sample from 1970Q1: log real GDP per worker. "
        rf"{df.country.nunique()} countries, {len(df)} observations, "
        rf"{df.period.min()}--{df.period.max()}."
    )
    print(sample_line, flush=True)
    FIGS.mkdir(exist_ok=True)
    fitted = []
    for spec in SPECS:
        res = fit_spec(df, spec)
        fig = FIGS / f"cycle_{spec['key']}.pdf"
        save_cycle_pdf(res, fig, spec["title"])
        rel = fig.relative_to(HERE).as_posix()
        fitted.append((spec, res, Path(rel)))
        aa = res.params.get("a")
        if spec["country_intercepts"] and isinstance(aa, dict):
            vals = np.array(list(aa.values()), float)
            print(
                f"  a_i mean={vals.mean():.4f}  min={vals.min():.4f}  "
                f"max={vals.max():.4f}  cycle mean="
                f"{np.mean([d['cycle'].mean() for d in res.filtered_probs.values()]):.4f}",
                flush=True,
            )
    TEX.write_text(_tex_report(sample_line, fitted), encoding="utf-8")
    compile_tex(TEX)
    print(f"\nWrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
