"""Common linear trend + RE intercepts on 2026-09-11 real GDP per worker."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOE = ROOT / "soe_sample"
CSV = HERE / "realGDP_nsa_empl.csv"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOE))

from panel_msar import PanelMSAR
from run_soe_spec_report import compile_tex, save_cycle_pdf, _tabular, _trend_note

OUT_PDF = HERE / "msar_re.pdf"
TEX = HERE / "msar_re.tex"
FIGS = HERE / "figs"
SAMPLE_CSV = HERE / "estimation_sample.csv"
YCOL = "realGDP_usd_pa_empl"

SPECS = [
    dict(
        key="common_ag",
        title="Common intercept and linear trend",
        country_intercepts=False,
        random_intercepts=False,
    ),
    dict(
        key="re_ai",
        title="Random-effects intercepts, common linear trend",
        country_intercepts=False,
        random_intercepts=True,
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
            "country",
            "country_name",
            "iso3",
            "year",
            "quarter",
            "time",
            "period",
            "t",
            YCOL,
            "y",
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
        r"\title{Panel MS-AR(1) with random-effects intercepts, full sample\\[0.4em]"
        r"\large Real GDP per worker (period-average USD)}",
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
        r"s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}), \\",
        r"a_i &\sim \mathcal{N}(\alpha,\omega^2) \quad \text{(random effects)}.",
        r"\end{align}",
        r"The outcome is log real GDP per worker (period-average USD) "
        r"from the 2026-09-11 quarterly panel. "
        r"Calendar time is taken from the period stamp $t$ (year-fraction). "
        r"The trend is linear in calendar time $t$ ($g$ per year, common across countries). "
        r"In the pooled specification $a_i\equiv a$. In the random-effects specification "
        r"the $a_i$ are i.i.d.\ Normal draws; $\alpha$ and $\omega$ are estimated by "
        r"maximum likelihood, integrating each country's Hamilton-filter likelihood "
        r"by 11-point Gauss--Hermite quadrature. Cycles are plotted using the "
        r"posterior mean of $a_i$. "
        r"Latent paths $s_{it}$ are country-specific. The regime dated $t$ governs "
        r"the transition from $z_t$ to $z_{t+1}$. $\sigma$ and $\rho$ switch with the "
        r"regime. The median $\mu$ is pinned at 0. $|\rho|<0.99$.",
        r"Standard errors (in parentheses) are delta-method from a numerical Hessian "
        r"on shared parameters. $\pi$ is the ergodic distribution of $\Pi$. "
        r"$E[z]$ is the long-run mean of the cycle implied by $\mu$, $\rho$, and $\Pi$.",
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
        random_intercepts=spec["random_intercepts"],
        two_step=False,
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
    df = load_panel()
    df.to_csv(SAMPLE_CSV, index=False)
    sample_line = (
        rf"2026-09-11 panel, full sample: log real GDP per worker. "
        rf"{df.country.nunique()} countries, {len(df)} observations, "
        rf"{df.period.min()}--{df.period.max()}."
    )
    print(sample_line, flush=True)
    print(f"Wrote {SAMPLE_CSV}", flush=True)
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
