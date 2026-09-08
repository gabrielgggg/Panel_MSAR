"""Fit six 3-regime SOE specs and compile a LaTeX report PDF."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from panel_msar import PanelMSAR
from prepare_soe_panel import load_panel

OUT_PDF = HERE / "soe_msar_specs.pdf"
TEX = HERE / "soe_msar_specs.tex"
FIGS = HERE / "figs"

SPECS = [
    dict(
        key="1_common_ag",
        title="Common intercept and linear trend",
        country_intercepts=False,
        country_trends=False,
        two_step=False,
        zero_mu=False,
        common_rho=False,
        compute_se=True,
    ),
    dict(
        key="2_ai_common_g",
        title="Country-specific intercepts, common linear trend",
        country_intercepts=True,
        country_trends=False,
        two_step=False,
        zero_mu=False,
        common_rho=False,
        compute_se=False,
    ),
    dict(
        key="3_ai_gi",
        title="Country-specific intercepts and linear trends",
        country_intercepts=True,
        country_trends=True,
        two_step=False,
        zero_mu=False,
        common_rho=False,
        compute_se=True,
    ),
    dict(
        key="4_cf",
        title="Country-specific CF filtered trend (25 years)",
        country_intercepts=False,
        country_trends=False,
        two_step="cf",
        zero_mu=False,
        common_rho=False,
        compute_se=True,
    ),
    dict(
        key="5_cf_zero_mu",
        title="CF filtered trend, all regime means restricted to 0",
        country_intercepts=False,
        country_trends=False,
        two_step="cf",
        zero_mu=True,
        common_rho=False,
        compute_se=True,
    ),
    dict(
        key="6_cf_common_rho",
        title="CF filtered trend, common $\\rho$",
        country_intercepts=False,
        country_trends=False,
        two_step="cf",
        zero_mu=False,
        common_rho=True,
        compute_se=True,
    ),
]


def _as1d(x, k):
    a = np.atleast_1d(x).astype(float)
    if a.size == 1:
        return np.full(k, float(a[0]))
    return a


def _se_at(arr, i):
    if arr is None:
        return None
    a = np.atleast_1d(arr)
    if a.size <= i:
        return None
    v = float(a[i])
    return v if np.isfinite(v) else None


def _tex_est(x):
    return f"{float(x):.4f}"


def _tex_se(se, pinned=False):
    if pinned:
        return r"---"
    if se is None or not np.isfinite(se):
        return ""
    return f"({float(se):.4f})"


def _cycle_colors(n):
    colors = [plt.colormaps["tab20"](i) for i in range(min(n, 20))]
    if n > 20:
        colors += [plt.colormaps["tab20b"](i) for i in range(min(n - 20, 20))]
    if n > 40:
        colors += [plt.colormaps["tab20c"](i) for i in range(min(n - 40, 20))]
    if n > 60:
        extra = plt.colormaps["Dark2"]
        colors += [extra(i) for i in range(n - 60)]
    return colors


def save_cycle_pdf(res, path, title):
    ids = [cid for cid in res.country_ids if cid in res.filtered_probs]
    n = len(ids)
    colors = _cycle_colors(n)
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    for i, cid in enumerate(ids):
        d = res.filtered_probs[cid]
        ax.plot(d["time"], d["cycle"], color=colors[i], lw=0.85, alpha=0.9)
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cycle (trend removed)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _pair_rows(name, vals, ses, k, pinned=None):
    est = [name] + [_tex_est(vals[s]) for s in range(k)]
    se = [""] + [
        _tex_se(
            _se_at(ses, s),
            pinned=bool(pinned[s]) if pinned is not None else False,
        )
        for s in range(k)
    ]
    return est, se


def _tabular(res):
    k = res.n_regimes
    mid = k // 2
    pr = res.params
    se = res.se_params
    mu = _as1d(pr["mu"], k)
    sig = _as1d(pr["sigma"], k)
    rho = _as1d(pr["rho"], k)
    P = np.asarray(pr["P"], dtype=float)
    se_mu = np.atleast_1d(se["mu"]) if se is not None else None
    se_sig = np.atleast_1d(se["sigma"]) if se is not None else None
    se_rho = np.atleast_1d(se["rho"]) if se is not None else None
    se_P = np.asarray(se["P"]) if se is not None and "P" in se else None
    pin_mu = np.zeros(k, dtype=bool)
    if res.zero_mu:
        pin_mu[:] = True
    else:
        pin_mu[mid] = True

    rows = []
    if res.common_rho:
        r = np.full(k, float(rho[0]))
        sr = np.full(k, np.nan)
        if se_rho is not None:
            sr[:] = _se_at(se_rho, 0)
        pairs = [
            (r"$\mu$", mu, se_mu, pin_mu),
            (r"$\sigma$", sig, se_sig, None),
            (r"$\rho$ (common)", r, sr, None),
        ]
    else:
        pairs = [
            (r"$\mu$", mu, se_mu, pin_mu),
            (r"$\sigma$", sig, se_sig, None),
            (r"$\rho$", rho, se_rho, None),
        ]
    for name, vals, ses, pin in pairs:
        rows.extend(_pair_rows(name, vals, ses, k, pin))
    for i in range(k):
        se_row = se_P[i] if se_P is not None else None
        rows.extend(_pair_rows(rf"$\Pi_{{{i}\cdot}}$", P[i], se_row, k))

    header = " & ".join([""] + [rf"({s})" for s in range(k)]) + r" \\"
    body = []
    for i, row in enumerate(rows):
        line = " & ".join(row) + r" \\"
        if i % 2 == 1:
            line += r" \addlinespace"
        body.append(line)
    col = "l" + "c" * k
    return (
        "\\begin{tabular}{" + col + "}\n"
        "\\toprule\n"
        + header
        + "\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}"
    )


def _trend_note(res):
    pr = res.params
    se = res.se_params or {}
    bits = []
    if res.two_step == "cf":
        bits.append(
            rf"Christiano--Fitzgerald low-pass trend: periods longer than "
            rf"{res.cf_high:.0f} observations ({res.cf_cutoff:g} years)."
        )
        return " ".join(bits)
    a, g = pr.get("a"), pr.get("g")
    if res.country_intercepts and isinstance(a, dict):
        aa = np.array(list(a.values()), dtype=float)
        bits.append(
            rf"Country intercepts $a_i$: mean {aa.mean():.3f}, "
            rf"min {aa.min():.3f}, max {aa.max():.3f}."
        )
    elif a is not None and not isinstance(a, dict):
        sa = se.get("a")
        extra = f" ({float(sa):.4f})" if sa is not None and np.isfinite(sa) else ""
        bits.append(rf"Common intercept $a={float(a):.4f}${extra}.")
    if res.country_trends and isinstance(g, dict):
        gg = np.array(list(g.values()), dtype=float)
        bits.append(
            rf"Country slopes $g_i$: mean {gg.mean():.4f}, "
            rf"min {gg.min():.4f}, max {gg.max():.4f}."
        )
    elif g is not None and not isinstance(g, dict):
        sg = se.get("g")
        extra = f" ({float(sg):.4f})" if sg is not None and np.isfinite(sg) else ""
        bits.append(rf"Common slope $g={float(g):.4f}${extra}.")
    return " ".join(bits)


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
        r"\title{Panel MS-AR(1) on SOE log real GDP per worker}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage",
        r"\section{Model}",
        r"Joint panel Markov-switching AR(1) around a trend, with three regimes:",
        r"\begin{align}",
        r"y_{it} &= \tau_{it} + z_{it}, \\",
        r"z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it}) + \rho(s_{it})\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}, \\",
        r"s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}).",
        r"\end{align}",
        r"Here $\tau_{it}$ is a (possibly country-specific) trend in calendar time; "
        r"the six specifications below differ in how $\tau_{it}$ is constructed. "
        r"Countries are independent given shared Markov parameters; latent paths "
        r"$s_{it}$ are country-specific. The regime dated $t$ governs the transition "
        r"from $z_t$ to $z_{t+1}$. $\sigma$ switches with the regime. Unless noted, "
        r"$\rho$ is regime-specific and the median $\mu$ is pinned at 0 after ordering. "
        r"The likelihood is a Hamilton filter per country, summed across the panel.",
        r"\subsection*{Trend assumptions}",
        r"\begin{enumerate}",
        r"\item Common intercept $a$ and common linear slope $g$ (joint MLE).",
        r"\item Country intercepts $a_i$ and common $g$ ($a_i$ profiled).",
        r"\item Country intercepts $a_i$ and slopes $g_i$ (both profiled).",
        r"\item Country-specific Christiano--Fitzgerald low-pass trend "
        r"(cycle $=$ oscillations of 2--100 quarters / 25 years), then MS-AR on the cycle.",
        r"\item Same CF trend, with every $\mu(s)=0$.",
        r"\item Same CF trend, with one $\rho$ common to all regimes.",
        r"\end{enumerate}",
        r"Sections 4--6 are two-step: the trend is not re-estimated jointly with the cycle.",
        r"Standard errors (in parentheses) are delta-method from a numerical Hessian on "
        r"shared parameters only. A dashed entry is a pinned coefficient.",
    ]
    for spec, res, fig in fitted:
        se_note = ""
        if res.se_params is None:
            se_note = " Standard errors omitted."
        parts += [
            r"\clearpage",
            rf"\section{{{spec['title']}}}",
            sample_line + rf" Fitted countries: {res.n_countries}. "
            rf"Observations: {res.nobs}. Log-likelihood: {res.loglik:.2f}."
            + se_note,
            _trend_note(res),
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Regime AR(1) and transition matrix. Columns are regimes.}",
            _tabular(res),
            r"\end{table}",
            r"\begin{figure}[h]",
            r"\centering",
            rf"\includegraphics[width=0.75\textwidth]{{{fig.as_posix()}}}",
            r"\caption{Country cycles after removing the section's trend.}",
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
        common_rho=spec["common_rho"],
        common_sigma=False,
        country_intercepts=spec["country_intercepts"],
        country_trends=spec["country_trends"],
        two_step=spec["two_step"],
        cf_cutoff=25,
        zero_mu=spec["zero_mu"],
        min_t=12,
    )
    res = mod.fit(
        df["country"],
        df["time"],
        df["y"],
        n_starts=8,
        maxiter=400,
        seed=1,
        compute_se=spec["compute_se"],
        store_filtered=True,
        verbose=verbose,
    )
    txt = HERE / f"output_spec{spec['key']}_SOE.txt"
    header = (
        f"SOE log real GDP per worker\n{spec['title']}\n"
        f"n_regimes=3  two_step={spec['two_step']!r}  "
        f"country_intercepts={spec['country_intercepts']}  "
        f"country_trends={spec['country_trends']}  "
        f"zero_mu={spec['zero_mu']}  common_rho={spec['common_rho']}  "
        f"cf_cutoff=25  compute_se={spec['compute_se']}\n\n"
    )
    txt.write_text(header + res.summary() + "\n", encoding="utf-8")
    print(res, flush=True)
    print(f"Wrote {txt}", flush=True)
    return res


def compile_tex(tex_path: Path):
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    for _ in range(2):
        subprocess.run(
            cmd,
            cwd=str(tex_path.parent),
            check=True,
        )


def main():
    df = load_panel()
    sample_line = (
        rf"SOE sample: log real GDP per worker. "
        rf"{df.country.nunique()} countries in the file, {len(df)} observations, "
        rf"{df.period.min()}--{df.period.max()}."
    )
    print(sample_line, flush=True)
    FIGS.mkdir(exist_ok=True)
    fitted = []
    for spec in SPECS:
        res = fit_spec(df, spec)
        fig = FIGS / f"cycle_{spec['key']}.pdf"
        save_cycle_pdf(res, fig, spec["title"])
        # graphicx path relative to the .tex file in HERE
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
    produced = HERE / "soe_msar_specs.pdf"
    print(f"\nWrote {produced}", flush=True)


if __name__ == "__main__":
    main()
