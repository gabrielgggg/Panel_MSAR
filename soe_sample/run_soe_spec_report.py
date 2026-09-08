"""Fit six 3-regime SOE specs and write a vector PDF (table + cycle plot each)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from panel_msar import PanelMSAR
from prepare_soe_panel import load_panel

OUT_PDF = HERE / "soe_msar_specs.pdf"

SPECS = [
    dict(
        key="1_common_ag",
        title="1. Common intercept and linear trend",
        country_intercepts=False,
        country_trends=False,
        two_step=False,
        zero_mu=False,
        common_rho=False,
    ),
    dict(
        key="2_ai_common_g",
        title="2. Country-specific intercepts, common linear trend",
        country_intercepts=True,
        country_trends=False,
        two_step=False,
        zero_mu=False,
        common_rho=False,
    ),
    dict(
        key="3_ai_gi",
        title="3. Country-specific intercepts and linear trends",
        country_intercepts=True,
        country_trends=True,
        two_step=False,
        zero_mu=False,
        common_rho=False,
    ),
    dict(
        key="4_cf",
        title="4. Country-specific CF filtered trend (25 years)",
        country_intercepts=False,
        country_trends=False,
        two_step="cf",
        zero_mu=False,
        common_rho=False,
    ),
    dict(
        key="5_cf_zero_mu",
        title="5. CF filtered trend, all regime means restricted to 0",
        country_intercepts=False,
        country_trends=False,
        two_step="cf",
        zero_mu=True,
        common_rho=False,
    ),
    dict(
        key="6_cf_common_rho",
        title="6. CF filtered trend, common rho",
        country_intercepts=False,
        country_trends=False,
        two_step="cf",
        zero_mu=False,
        common_rho=True,
    ),
]


def _as1d(x, k):
    a = np.atleast_1d(x).astype(float)
    if a.size == 1:
        return np.full(k, float(a[0]))
    return a


def _fmt_est(x):
    return f"{float(x):.4f}"


def _fmt_se(se, pinned=False):
    if pinned:
        return "(—)"
    if se is None or not np.isfinite(se):
        return ""
    return f"({float(se):.4f})"


def _se_at(arr, i):
    if arr is None:
        return None
    a = np.atleast_1d(arr)
    if a.size <= i:
        return None
    return float(a[i])


def _ag_line(res):
    pr = res.params
    bits = []
    if res.two_step == "cf":
        bits.append(
            f"CF low-pass: periods longer than {res.cf_high:.0f} obs "
            f"({res.cf_cutoff:g} years)"
        )
        return "  ".join(bits)
    a = pr.get("a")
    g = pr.get("g")
    se = res.se_params or {}
    if res.country_intercepts and isinstance(a, dict):
        aa = np.array(list(a.values()), dtype=float)
        bits.append(f"a_i  mean={aa.mean():.4f}  min={aa.min():.4f}  max={aa.max():.4f}")
    elif a is not None and not isinstance(a, dict):
        sa = se.get("a")
        extra = f"  {_fmt_se(sa)}" if sa is not None else ""
        bits.append(f"a  {_fmt_est(a)}{extra}")
    if res.country_trends and isinstance(g, dict):
        gg = np.array(list(g.values()), dtype=float)
        bits.append(f"g_i  mean={gg.mean():.4f}  min={gg.min():.4f}  max={gg.max():.4f}")
    elif g is not None and not isinstance(g, dict):
        sg = se.get("g")
        extra = f"  {_fmt_se(sg)}" if sg is not None else ""
        bits.append(f"g  {_fmt_est(g)}{extra}")
    return "     ".join(bits)


def _regime_table_rows(res):
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
    header = [""] + [f"regime {s}" for s in range(k)]
    rows.append(header)

    def add_pair(name, vals, ses, pinned=None):
        est = [name] + [_fmt_est(vals[s]) for s in range(k)]
        se_row = [""] + [
            _fmt_se(_se_at(ses, s), pinned=bool(pinned[s]) if pinned is not None else False)
            for s in range(k)
        ]
        rows.append(est)
        rows.append(se_row)

    add_pair("mu", mu, se_mu, pin_mu)
    add_pair("sigma", sig, se_sig)
    if res.common_rho:
        r = np.full(k, float(rho[0]))
        sr = np.full(k, _se_at(se_rho, 0) if se_rho is not None else np.nan)
        add_pair("rho (common)", r, sr)
    else:
        add_pair("rho", rho, se_rho)
    for i in range(k):
        se_row = se_P[i] if se_P is not None else None
        add_pair(f"Pi from {i}", P[i], se_row)
    return rows


def _table_page(spec, res, sample_line):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(spec["title"], fontsize=13, fontweight="bold", y=0.96)
    meta = (
        f"{sample_line}\n"
        f"Regimes: {res.n_regimes}    countries: {res.n_countries}    "
        f"obs: {res.nobs}    log-likelihood: {res.loglik:.2f}\n"
        f"Converged: {res.success}    {res.message}\n"
        f"{_ag_line(res)}"
    )
    fig.text(0.08, 0.88, meta, va="top", ha="left", fontsize=9, family="monospace")
    rows = _regime_table_rows(res)
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.62])
    ax.axis("off")
    ax.set_title("Regime AR(1) and transition matrix (SEs in parentheses)", fontsize=10, pad=8)
    table = ax.table(
        cellText=rows,
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
        if c == 0:
            cell.set_text_props(ha="left")
    return fig


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


def _cycle_page(spec, res):
    ids = [cid for cid in res.country_ids if cid in res.filtered_probs]
    n = len(ids)
    colors = _cycle_colors(n)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for i, cid in enumerate(ids):
        d = res.filtered_probs[cid]
        ax.plot(d["time"], d["cycle"], color=colors[i], lw=0.9, alpha=0.9, label=str(cid))
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("cycle (trend removed)")
    ax.set_title(spec["title"] + " — detrended series")
    ax.grid(True, alpha=0.3)
    ax.legend(
        ncol=3, fontsize=6, loc="center left",
        bbox_to_anchor=(1.01, 0.5), frameon=False, borderaxespad=0,
    )
    fig.tight_layout()
    return fig


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
        compute_se=True,
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
        f"cf_cutoff=25  compute_se=True\n\n"
    )
    txt.write_text(header + res.summary() + "\n", encoding="utf-8")
    print(res, flush=True)
    print(f"Wrote {txt}", flush=True)
    return res


def main():
    df = load_panel()
    sample_line = (
        f"SOE sample: log real GDP per worker    "
        f"countries in file={df.country.nunique()}    obs={len(df)}    "
        f"{df.period.min()}–{df.period.max()}"
    )
    print(sample_line, flush=True)
    fitted = []
    for spec in SPECS:
        res = fit_spec(df, spec)
        fitted.append((spec, res))
    with PdfPages(OUT_PDF) as pdf:
        for spec, res in fitted:
            fig = _table_page(spec, res, sample_line)
            pdf.savefig(fig)
            plt.close(fig)
            fig = _cycle_page(spec, res)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"\nWrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
