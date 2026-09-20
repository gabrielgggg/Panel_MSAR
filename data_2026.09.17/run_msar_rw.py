"""Two-step: common local-level RW trend, then RE a_i + Barro catch-up, rho<=0.95."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

OUT_PDF = HERE / "msar_rw_lam99.pdf"
TEX = HERE / "msar_rw_lam99.tex"
FIGS = HERE / "figs"
SAMPLE_CSV = HERE / "rw_sample.csv"
TAU_CSV = HERE / "rw_tau.csv"
RHO_MAX = 0.99
YEAR_MIN = 1987
LAM_FIXED = 0.99

SPEC = dict(
    key="rw_lam99",
    title="Common RW trend removed; RE intercepts plus catch-up, lambda=0.99",
    random_intercepts=True,
    convergence=True,
    lambda_value=LAM_FIXED,
)


def subtract_tau(df: pd.DataFrame, times, tau):
    lookup = pd.Series(np.asarray(tau, dtype=float), index=np.asarray(times, dtype=float))
    out = df.copy()
    out["tau"] = out["time"].map(lookup)
    if out["tau"].isna().any():
        raise ValueError("Some country-times have no common tau_t.")
    out["y_raw"] = out["y"].astype(float)
    out["y"] = out["y_raw"] - out["tau"].astype(float)
    return out


def plot_tau_spaghetti(df: pd.DataFrame, times, tau, path: Path):
    """Raw country log levels (faint) and common tau_t (thick)."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ycol = "y_raw" if "y_raw" in df.columns else "y"
    for _, g in df.groupby("country"):
        ax.plot(
            g["time"].to_numpy(),
            g[ycol].to_numpy(),
            color="0.25",
            lw=0.55,
            alpha=0.18,
            solid_capstyle="butt",
        )
    ax.plot(
        np.asarray(times, dtype=float),
        np.asarray(tau, dtype=float),
        color="0.0",
        lw=2.8,
        solid_capstyle="round",
        zorder=5,
        label=r"common global trend $\tau_t$ (mean growth of continuing countries)",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("log real GDP per worker")
    ax.set_title("Common trend and country series")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _tex_preamble(n_bar, tau_info):
    return [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,graphicx,setspace,caption,hyperref}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathpazo}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\captionsetup{font=small,skip=6pt}",
        r"\title{Panel MS-AR(1) after a common random-walk trend\\[0.4em]"
        r"\large Seasonally adjusted real GDP per worker}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Model}",
        r"Two-step. Step 1 builds a \emph{common} trend from average growth of "
        r"countries observed in consecutive quarters (not from the level mean, "
        r"which falls when poorer countries enter). "
        rf"Log seasonally adjusted real GDP per worker, 2026-09-17 vintage, "
        rf"from {YEAR_MIN}Q1 onward.",
        r"\begin{align}",
        r"\Delta\tau_t &= \frac{1}{n_t^{\mathrm{cont}}}\sum_{i\in C_t\cap C_{t-1}} (y_{it}-y_{i,t-1}), \\",
        r"\tau_t &= \tau_{t_0}+\sum_{s=t_0+1}^{t}\Delta\tau_s.",
        r"\end{align}",
        r"The level of $\tau$ is pinned to the balanced-core mean at $t_0$. "
        rf"Mean $\Delta\tau$ is {tau_info['mu']:.4f} per quarter "
        rf"({tau_info.get('n_core', 0)} countries present in every quarter). "
        r"Step 2 is estimated on $y_{it}^\ast=y_{it}-\hat\tau_t$, with no extra $gt$:",
        r"\begin{align}",
        r"y_{it}^\ast &= a_i + b_i\,\lambda^{t-T_{i0}} + z_{it}, \\",
        r"a_i &\sim \mathcal{N}(\alpha,\omega^2),\qquad "
        r"b_i \sim \mathcal{N}(0,\omega_b^2) \ \text{independent}, \\",
        r"z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it}) + \rho(s_{it})\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}.",
        r"\end{align}",
        rf"$\lambda={LAM_FIXED}$ is fixed. "
        rf"$|\rho|<{RHO_MAX}$. "
        r"Three regimes; median $\mu$ pinned at 0; $\sigma$ and $\rho$ switch. "
        r"$T_{i0}$ is country $i$'s first observation after filters. "
        r"Random effects: 7$\times$7 Gauss--Hermite. "
        r"Plotted cycles use posterior-mean $a_i$ and $b_i$. "
        r"The common trend $\tau_t$ is discarded for the structural MS-AR. "
        r"Standard errors ignore step-1 uncertainty.",
        r"\begin{figure}[h]",
        r"\centering",
        rf"\includegraphics[width=0.88\textwidth]{{{n_bar.as_posix()}}}",
        r"\caption{Faint lines: country log real GDP per worker. "
        r"Thick line: common $\tau_t$ from average growth of continuing countries "
        r"(new entrants do not shift the level).}",
        r"\end{figure}",
    ]


def _tex_report(sample_line, fitted, tau_fig, tau_info):
    parts = _tex_preamble(tau_fig, tau_info)
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
            r"\caption{Country cycles after removing $\tau_t$, $a_i$, and $b_i\lambda^{t-T_{i0}}$.}",
            r"\end{figure}",
        ]
    parts.append(r"\end{document}")
    return "\n".join(parts) + "\n"


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
        f"continuing {int(tau_info['n_delta'][1:].min())}--"
        f"{int(tau_info['n_delta'][1:].max())}, "
        f"core {tau_info['n_core']}, mean dtau={tau_info['mu']:.5f}",
        flush=True,
    )
    df = subtract_tau(raw, grid, tau_info["tau"])
    df.to_csv(SAMPLE_CSV, index=False)
    pd.DataFrame({
        "time": grid,
        "ybar": tau_info["ybar"],
        "tau": tau_info["tau"],
        "n": tau_info["n"],
        "n_delta": tau_info["n_delta"],
    }).to_csv(TAU_CSV, index=False)
    print(f"Wrote {SAMPLE_CSV} and {TAU_CSV}", flush=True)
    FIGS.mkdir(exist_ok=True)
    tau_fig = FIGS / "tau_rw.pdf"
    plot_tau_spaghetti(df, grid, tau_info["tau"], tau_fig)
    sample_line = (
        rf"2026-09-17 SA panel, $y_{{it}}-\hat\tau_t$, from {YEAR_MIN}. "
        rf"{df.country.nunique()} countries, {len(df)} observations, "
        rf"{df.period.min()}--{df.period.max()}. "
        rf"$|\rho|<{RHO_MAX}$, $\lambda={LAM_FIXED}$ fixed."
    )
    print(sample_line, flush=True)
    spec = SPEC
    print("\n" + "=" * 72, flush=True)
    print(spec["title"], flush=True)
    print("=" * 72, flush=True)
    mod = PanelMSAR(
        n_regimes=3,
        common_rho=False,
        common_sigma=False,
        random_intercepts=True,
        convergence=True,
        lambda_value=LAM_FIXED,
        lambda_max=0.995,
        include_trend=False,
        zero_mu=False,
        min_t=12,
        rho_max=RHO_MAX,
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
        verbose=True,
    )
    print(res, flush=True)
    fig = FIGS / f"cycle_{spec['key']}.pdf"
    save_cycle_pdf(res, fig, spec["title"])
    rel = fig.relative_to(HERE)
    tau_rel = tau_fig.relative_to(HERE)
    TEX.write_text(
        _tex_report(sample_line, [(spec, res, rel)], tau_rel, tau_info),
        encoding="utf-8",
    )
    compile_tex(TEX)
    print(f"\nWrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
