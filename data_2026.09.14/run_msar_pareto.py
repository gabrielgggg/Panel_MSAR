"""Common Q dummies: pooled trend vs Pareto Type II random intercepts."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOE = ROOT / "soe_sample"
CSV = HERE / "realGDP_nsa_empl_s3_both.csv"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOE))

from panel_msar import PanelMSAR
from run_soe_spec_report import compile_tex, save_cycle_pdf, _tabular, _trend_note

OUT_PDF = HERE / "msar_pareto.pdf"
TEX = HERE / "msar_pareto.tex"
FIGS = HERE / "figs"
YCOL = "realGDP_usd_pa_empl"

SPECS = [
    dict(
        key="common_ag_q",
        title="Common intercept, linear trend, and quarterly dummies",
        random_intercepts=False,
        re_family="normal",
    ),
    dict(
        key="pareto_ai_q",
        title="Pareto RE intercepts, common trend and quarterly dummies",
        random_intercepts=True,
        re_family="pareto",
    ),
]


def load_panel(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    raw = pd.read_csv(path)
    need = {"iso3", "t", YCOL}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    df = raw.copy()
    df["iso3"] = df["iso3"].astype(str).str.strip()
    parsed = df["t"].astype(str).str.extract(
        r"(?P<year>\d{4})q(?P<quarter>[1-4])", flags=re.I
    )
    df["year"] = pd.to_numeric(parsed["year"], errors="coerce")
    df["quarter"] = pd.to_numeric(parsed["quarter"], errors="coerce")
    df["time"] = df["year"] + (df["quarter"] - 1) / 4.0
    df["period"] = [
        f"{int(y)}-Q{int(q)}" if pd.notna(y) and pd.notna(q) else ""
        for y, q in zip(df["year"], df["quarter"])
    ]
    df[YCOL] = pd.to_numeric(df[YCOL], errors="coerce")
    pos = df[YCOL].notna() & (df[YCOL] > 0)
    df = df.loc[pos].copy()
    df["y"] = np.log(df[YCOL].astype(float))
    if "country" in df.columns:
        df["country_name"] = df["country"].astype(str)
    else:
        df["country_name"] = df["iso3"]
    df["country"] = df["iso3"]
    df = df.dropna(subset=["country", "time", "y"])
    df = df.sort_values(["country", "time"], kind="mergesort").reset_index(drop=True)
    dup = df.duplicated(["country", "time"])
    if dup.any():
        raise ValueError(
            f"{int(dup.sum())} duplicate country-time rows after cleaning."
        )
    keep = [
        c
        for c in (
            "country", "country_name", "iso3", "year", "quarter",
            "time", "period", "t", YCOL, "y",
        )
        if c in df.columns
    ]
    return df.loc[:, keep]


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
        r"\title{Panel MS-AR(1): common trend vs Pareto random intercepts\\[0.4em]"
        r"\large Quarterly dummies, real GDP per worker}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Model}",
        r"Joint panel Markov-switching AR(1) around a linear trend with "
        r"common quarterly dummies (Q1 omitted):",
        r"\begin{align}",
        r"y_{it} &= a_i + g\, t + d_2 Q2_t + d_3 Q3_t + d_4 Q4_t + z_{it}, \\",
        r"z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it}) + \rho(s_{it})\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}, \\",
        r"s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}).",
        r"\end{align}",
        r"Specification 1 sets $a_i\equiv a$. Specification 2 draws "
        r"$a_i$ from a shifted Type II Pareto (Lomax) on $[m,\infty)$, "
        r"with location $m$, scale $\sigma$, and shape $\alpha$: "
        r"$P(a_i>x)=(1+(x-m)/\sigma)^{-\alpha}$ for $x\ge m$. "
        r"The country likelihood integrates $a_i$ by 15-point Gauss--Legendre "
        r"on the probability-integral-transform scale. "
        r"The slope $g$ and the quarterly dummies are common. $|\rho|<0.99$. "
        r"Median $\mu$ pinned at 0. Cycles use posterior-mean $a_i$. "
        r"Standard errors are delta-method from a numerical Hessian.",
    ]
    for spec, res, fig in fitted:
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
            r"\caption{Country cycles after removing the linear trend and quarterly means.}",
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
        country_intercepts=False,
        country_trends=False,
        random_intercepts=spec["random_intercepts"],
        two_step=False,
        zero_mu=False,
        min_t=12,
        rho_max=0.99,
        quarter_dummies=True,
        random_seasonals=False,
        re_family=spec["re_family"],
    )
    res = mod.fit(
        df["country"],
        df["time"],
        df["y"],
        n_starts=3,
        maxiter=400,
        seed=1,
        compute_se=True,
        store_filtered=True,
        verbose=verbose,
    )
    print(res, flush=True)
    return res


def main():
    df = load_panel()
    sample_line = (
        rf"2026-09-14 panel, full sample: log real GDP per worker. "
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
    TEX.write_text(_tex_report(sample_line, fitted), encoding="utf-8")
    compile_tex(TEX)
    print(f"\nWrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
