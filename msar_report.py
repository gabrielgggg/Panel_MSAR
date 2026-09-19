"""LaTeX helpers for panel MS-AR reports."""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    path = Path(path)
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
    if "pi" in pr:
        se_pi = np.atleast_1d(se["pi"]) if se is not None and se.get("pi") is not None else None
        rows.extend(_pair_rows(r"$\pi$", _as1d(pr["pi"], k), se_pi, k))

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
    a, g = pr.get("a"), pr.get("g")
    if getattr(res, "convergence", False):
        al = pr.get("alpha", pr.get("a_bar"))
        lam = pr.get("lambda")
        om = pr.get("omega")
        omb = pr.get("omega_b")
        sa = se.get("alpha")
        sl = se.get("lambda")
        so = se.get("omega")
        sob = se.get("omega_b")
        extra_a = f" ({float(sa):.4f})" if sa is not None and np.isfinite(float(sa)) else ""
        extra_l = f" ({float(sl):.4f})" if sl is not None and np.isfinite(float(sl)) else ""
        extra_o = f" ({float(so):.4f})" if so is not None and np.isfinite(float(so)) else ""
        extra_b = f" ({float(sob):.4f})" if sob is not None and np.isfinite(float(sob)) else ""
        lam_txt = rf"$\lambda={float(lam):.4f}$"
        if getattr(res, "lambda_fixed", None) is not None:
            lam_txt += r" (fixed)"
        elif extra_l:
            lam_txt += extra_l
        eq = (
            r"$y_{it}=a_i + b_i\lambda^{t-T_{i0}}+z_{it}$"
            if not getattr(res, "include_trend", True)
            else r"$y_{it}=a_i + g t + b_i\lambda^{t-T_{i0}}+z_{it}$"
        )
        bits.append(
            rf"Permanent RE intercepts and catch-up "
            + eq +
            rf" with $\alpha={float(al):.4f}${extra_a}, "
            rf"$\omega={float(om):.4f}${extra_o}, "
            + lam_txt + rf", $\omega_b={float(omb):.4f}${extra_b}."
        )
        if isinstance(a, dict):
            aa = np.array(list(a.values()), dtype=float)
            bits.append(
                rf"Posterior-mean $a_i$: mean {aa.mean():.3f}, "
                rf"min {aa.min():.3f}, max {aa.max():.3f}."
            )
        bb = pr.get("b")
        if isinstance(bb, dict) and bb:
            bv = np.array(list(bb.values()), dtype=float)
            bits.append(
                rf"Posterior-mean $b_i$: mean {bv.mean():.3f}, "
                rf"min {bv.min():.3f}, max {bv.max():.3f}."
            )
    elif getattr(res, "random_intercepts", False):
        al, om = pr.get("alpha"), pr.get("omega")
        sa = se.get("alpha")
        so = se.get("omega")
        extra_a = f" ({float(sa):.4f})" if sa is not None and np.isfinite(float(sa)) else ""
        extra_o = f" ({float(so):.4f})" if so is not None and np.isfinite(float(so)) else ""
        bits.append(
            rf"Random intercepts $a_i\sim N(\alpha,\omega^2)$ with "
            rf"$\alpha={float(al):.4f}${extra_a}, $\omega={float(om):.4f}${extra_o}."
        )
        if isinstance(a, dict):
            aa = np.array(list(a.values()), dtype=float)
            bits.append(
                rf"Posterior-mean $a_i$: mean {aa.mean():.3f}, "
                rf"min {aa.min():.3f}, max {aa.max():.3f}."
            )
    elif a is not None and not isinstance(a, dict):
        sa = se.get("a")
        extra = f" ({float(sa):.4f})" if sa is not None and np.isfinite(sa) else ""
        bits.append(rf"Common intercept $a={float(a):.4f}${extra}.")
    if getattr(res, "include_trend", True) and g is not None and not isinstance(g, dict):
        sg = se.get("g")
        extra = f" ({float(sg):.4f})" if sg is not None and np.isfinite(sg) else ""
        bits.append(rf"Common slope $g={float(g):.4f}${extra}.")
    return " ".join(bits)


def compile_tex(tex_path: Path):
    tex_path = Path(tex_path)
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    for _ in range(2):
        subprocess.run(cmd, cwd=str(tex_path.parent), check=True)
