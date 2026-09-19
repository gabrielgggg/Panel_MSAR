"""US-gap panel: RE intercepts, then Barro catch-up, no common gt."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from panel_msar import BARRO_LAMBDA, PanelMSAR
from msar_report import compile_tex, save_cycle_pdf, _tabular, _trend_note
from run_msar_re import load_panel

OUT_PDF = HERE / "msar_usgap.pdf"
TEX = HERE / "msar_usgap.tex"
FIGS = HERE / "figs"
SAMPLE_CSV = HERE / "usgap_sample.csv"
US_ISO = "USA"

SPECS = [
    dict(
        key="gap_re",
        title="Log gap to the US: random intercepts, no trend",
        random_intercepts=True,
        convergence=False,
        lambda_value=None,
    ),
    dict(
        key="gap_barro",
        title="Log gap to the US: RE intercepts plus Barro catch-up",
        random_intercepts=True,
        convergence=True,
        lambda_value=BARRO_LAMBDA,
    ),
]


def make_us_gap(df: pd.DataFrame) -> pd.DataFrame:
    us = df.loc[df["country"] == US_ISO, ["time", "y"]].rename(columns={"y": "y_us"})
    if us.empty:
        raise ValueError(f"No {US_ISO} rows in the panel; cannot form the US gap.")
    us = us.drop_duplicates("time")
    out = df.merge(us, on="time", how="inner")
    out = out.loc[out["country"] != US_ISO].copy()
    out["y"] = out["y"].astype(float) - out["y_us"].astype(float)
    out = out.dropna(subset=["country", "time", "y"])
    out = out.sort_values(["country", "time"], kind="mergesort").reset_index(drop=True)
    return out


def _tex_preamble():
    return [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,graphicx,setspace,caption,hyperref}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathpazo}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\captionsetup{font=small,skip=6pt}",
        r"\title{Panel MS-AR(1) for the log gap to the United States\\[0.4em]"
        r"\large Seasonally adjusted real GDP per worker}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Model}",
        r"The outcome is the log gap to the US in the same calendar quarter,",
        r"\begin{equation}",
        r"d_{it}=y_{it}-y_{\mathrm{US},t},",
        r"\end{equation}",
        r"where $y$ is log seasonally adjusted real GDP per worker "
        r"(period-average USD) from the 2026-09-17 panel. "
        r"The United States is the numeraire and is dropped from the panel. "
        r"A country-quarter is kept only if both that country and the US are observed. "
        r"There is no common time trend: the US path already removes a shared $gt$.",
        r"Joint panel Markov-switching AR(1) with three regimes:",
        r"\begin{align}",
        r"d_{it} &= a_i + z_{it}, \\",
        r"z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it}) + \rho(s_{it})\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}, \\",
        r"s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}), \\",
        r"a_i &\sim \mathcal{N}(\alpha,\omega^2).",
        r"\end{align}",
        r"Spec 1 is this model. Spec 2 keeps the permanent gap and adds a decaying "
        r"entry deviation from that gap,",
        r"\begin{align}",
        r"d_{it} &= a_i + b_i\,\lambda^{t-T_{i0}} + z_{it}, \\",
        r"b_i &\sim \mathcal{N}(0,\omega_b^2) \ \text{independent of } a_i,",
        r"\end{align}",
        rf"with $\lambda={BARRO_LAMBDA}$ fixed (Barro 2\% per year). "
        r"$T_{i0}$ is country $i$'s first observation. "
        r"The long-run gap is $a_i$. Random effects are integrated by "
        r"Gauss--Hermite quadrature (11-point in spec 1; $7\times 7$ in spec 2). "
        r"Plotted cycles use posterior-mean $a_i$ (and $b_i$ in spec 2). "
        r"Latent paths $s_{it}$ are country-specific. $\sigma$ and $\rho$ switch. "
        r"The median $\mu$ is pinned at 0. $|\rho|<0.99$.",
        r"Standard errors (in parentheses) are delta-method from a numerical Hessian. "
        r"$\pi$ is the ergodic distribution of $\Pi$. $E[z]$ is the long-run mean of the cycle.",
    ]


def _tex_spec_section(sample_line, spec, res, fig):
    se_note = ""
    if res.se_params is None:
        se_note = " Standard errors omitted."
    ez = res.params.get("Ez")
    se = res.se_params or {}
    se_ez = se.get("Ez")
    ez_line = ""
    if ez is not None and np.isfinite(ez):
        extra = ""
        if se_ez is not None and np.isfinite(float(se_ez)):
            extra = f" ({float(se_ez):.4f})"
        ez_line = rf" Ergodic mean of the cycle $E[z]={float(ez):.4f}${extra}."
    return [
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
        r"\caption{Country cycles in the US gap after removing the deterministic path.}",
        r"\end{figure}",
    ]


def _tex_report(sample_line, fitted):
    parts = _tex_preamble()
    for spec, res, fig in fitted:
        parts += _tex_spec_section(sample_line, spec, res, fig)
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
        random_intercepts=spec["random_intercepts"],
        convergence=spec.get("convergence", False),
        lambda_value=spec.get("lambda_value"),
        lambda_max=0.985,
        include_trend=False,
        zero_mu=False,
        min_t=12,
        rho_max=0.99,
    )
    res = mod.fit(
        df["country"],
        df["time"],
        df["y"],
        n_starts=4,
        maxiter=400,
        seed=1,
        compute_se=True,
        store_filtered=True,
        verbose=verbose,
    )
    print(res, flush=True)
    return res


def main():
    raw = load_panel()
    df = make_us_gap(raw)
    df.to_csv(SAMPLE_CSV, index=False)
    print(f"Wrote {SAMPLE_CSV}", flush=True)
    sample_line = (
        rf"2026-09-17 SA panel, log gap to the US. "
        rf"{df.country.nunique()} countries (USA dropped), {len(df)} observations, "
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
    TEX.write_text(_tex_report(sample_line, fitted), encoding="utf-8")
    compile_tex(TEX)
    print(f"\nWrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
