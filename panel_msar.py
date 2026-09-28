"""
Two-step panel Markov-switching AR(1).

Step 1. Pooled OLS of y on a constant and calendar time t. The residual
e = y - a - g t satisfies sum(e) = 0 and sum(e * t) = 0 on the estimation
sample, so the cycle has no pooled level and no pooled slope.

Step 2. MS-AR(1) on e only. a and g are not likelihood parameters.

    s_{i,t+1} ~ Markov(Pi | s_it)
    z_{i,t+1} = (1 - rho(s_{i,t+1})) * mu(s_{i,t+1})
                + rho(s_{i,t+1}) * z_it
                + sigma(s_{i,t+1}) * eps

The regime dated t is the one that produced z_t. sigma switches with the
regime. One rho may be shared (common_rho=True) or each regime has its own.
E[z] = 0: the last regime mean, before ordering, is solved from that
restriction. After estimation, regimes are ordered by mu.

Countries are independent given shared parameters. The panel may be
unbalanced. Time is numeric calendar time on a common origin. Consecutive
rows are one AR(1) step. g is per unit of the time variable.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return deco


LOG2PI = np.log(2.0 * np.pi)
_NUMBA_WARMED = False
# |rho| < RHO_MAX. Unconstrained parameter is artanh(rho / RHO_MAX).
RHO_MAX = 0.995


def _rho_from_u(u, rho_max=RHO_MAX):
    return float(rho_max) * np.tanh(np.asarray(u, dtype=float))


def _u_from_rho(r, rho_max=RHO_MAX):
    cap = float(rho_max)
    x = np.clip(np.asarray(r, dtype=float) / cap, -0.999999, 0.999999)
    return np.arctanh(x)


def _softmax_rows(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def _stationary_probs(P, tol=1e-12):
    k = P.shape[0]
    A = P.T - np.eye(k)
    A[-1] = 1.0
    b = np.zeros(k)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pi = np.ones(k) / k
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    if s < tol:
        return np.ones(k) / k
    return pi / s


def ergodic_cycle_mean(mu, rho, P):
    """Stationary distribution of s and long-run E[z].

    Timing matches the filter: s' is drawn from s, then z' is drawn under s'.
    nu_j = E[z | s = j] solves
        pi_j * nu_j - rho_j * sum_i pi_i * P_{ij} * nu_i
            = pi_j * (1 - rho_j) * mu_j.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float)).reshape(-1)
    rho = np.atleast_1d(np.asarray(rho, dtype=float)).reshape(-1)
    P = np.asarray(P, dtype=float)
    k = int(P.shape[0])
    if rho.size == 1:
        rho = np.full(k, float(rho[0]))
    if mu.size == 1:
        mu = np.full(k, float(mu[0]))
    pi = _stationary_probs(P)
    # A[j, i] = pi[j] * 1{i=j} - rho[j] * pi[i] * P[i, j]
    A = np.diag(pi) - rho[:, None] * (pi[None, :] * P.T)
    b = pi * (1.0 - rho) * mu
    nu = None
    if np.linalg.cond(A) < 1e12:
        try:
            nu = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            nu = None
    if nu is None or not np.all(np.isfinite(nu)):
        return pi, float(np.dot(pi, mu))
    return pi, float(np.dot(pi, nu))


def _as_1d(name, x):
    if x is None:
        raise TypeError(f"{name} is required (got None).")
    arr = np.asarray(x)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be a 1-d sequence, got a scalar.")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}.")
    return arr


def _as_numeric_1d(name, x):
    arr = _as_1d(name, x)
    try:
        out = arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be numeric. Could not convert to float: {exc}"
        ) from exc
    return out


def _datetime_to_yearfrac(values):
    """Map dates to year + (month-1)/12 so quarter spacing is exactly 0.25."""
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(values)))
    return np.asarray(idx.year + (idx.month - 1) / 12.0, dtype=np.float64)


def _period_to_yearfrac(idx):
    """Map pandas Periods to a linear year scale (g then has annual units)."""
    idx = pd.PeriodIndex(idx)
    freq = (idx.freqstr or "").upper()
    year = np.asarray(idx.year, dtype=np.float64)
    if freq.startswith("A") or freq.startswith("Y"):
        return year
    if freq.startswith("Q"):
        return year + (np.asarray(idx.quarter, dtype=np.float64) - 1.0) / 4.0
    if freq.startswith("M"):
        return year + (np.asarray(idx.month, dtype=np.float64) - 1.0) / 12.0
    ts = idx.to_timestamp()
    return np.asarray(ts.year + (ts.month - 1) / 12.0, dtype=np.float64)


def _coerce_time(name, x):
    """Calendar time -> float. Datetimes/periods become year-fractions.

    Consecutive observations three months apart then have dt = 0.25, so g
    is per year and rho is per observation. Integer period indexes (Stata
    %tq, 0,1,2,...) are left as-is: g is then per period.
    """
    if isinstance(x, pd.PeriodIndex):
        return _period_to_yearfrac(x)
    if isinstance(x, pd.DatetimeIndex):
        return _datetime_to_yearfrac(x)
    if isinstance(x, pd.Series):
        if str(x.dtype).startswith("period"):
            return _period_to_yearfrac(pd.PeriodIndex(x))
        if pd.api.types.is_datetime64_any_dtype(x):
            return _datetime_to_yearfrac(x)
        x = x.to_numpy()
    arr = _as_1d(name, x)
    if arr.size == 0:
        return arr.astype(np.float64)
    if np.issubdtype(arr.dtype, np.datetime64):
        return _datetime_to_yearfrac(arr)
    if arr.dtype == object:
        first = arr[0]
        if isinstance(first, pd.Period):
            return _period_to_yearfrac(arr)
        if isinstance(first, (pd.Timestamp, np.datetime64)):
            return _datetime_to_yearfrac(arr)
        if hasattr(first, "year") and hasattr(first, "month") and not isinstance(
            first, (bytes, str)
        ):
            return _datetime_to_yearfrac(arr)
    return _as_numeric_1d(name, arr)


def _cell_est(x, width=12):
    return f"{float(x):{width}.4f}"


def _cell_se(se, width=12):
    if se is None or not np.isfinite(se):
        return " " * width
    return f"{'(' + f'{float(se):.4f}' + ')':>{width}}"


@njit(cache=True)
def _logsumexp_nb(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


@njit(cache=True)
def _country_ll_nb(z, rho, mu, sig, P, pi0):
    T = z.shape[0]
    k = mu.shape[0]
    if T < 2:
        return -1e10
    logP = np.log(np.clip(P, 1e-12, 1.0))
    log_sig = np.log(sig)
    var0 = sig ** 2 / np.maximum(1.0 - rho ** 2, 1e-8)
    sd0 = np.sqrt(var0)
    log_f0 = -0.5 * LOG2PI - np.log(sd0) - 0.5 * ((z[0] - mu) / sd0) ** 2
    log_joint = np.log(np.clip(pi0, 1e-12, 1.0)) + log_f0
    log_p = _logsumexp_nb(log_joint)
    ll = log_p
    log_filt = log_joint - log_p
    one_m_rho = 1.0 - rho
    log_num = np.empty(k)
    log_pred = np.empty(k)
    acc = np.empty(k)
    for t in range(T - 1):
        zt = z[t]
        ztp = z[t + 1]
        for sp in range(k):
            for s in range(k):
                acc[s] = log_filt[s] + logP[s, sp]
            log_pred[sp] = _logsumexp_nb(acc)
        for sp in range(k):
            resid = (ztp - (mu[sp] * one_m_rho[sp] + rho[sp] * zt)) / sig[sp]
            log_num[sp] = (
                log_pred[sp] - 0.5 * LOG2PI - log_sig[sp] - 0.5 * resid * resid
            )
        log_p = _logsumexp_nb(log_num)
        ll += log_p
        for s in range(k):
            log_filt[s] = log_num[s] - log_p
    return ll


@njit(cache=True)
def _panel_ll_nb(zcat, lengths, offsets, rho, mu, sig, P, pi0):
    ll = 0.0
    n = lengths.shape[0]
    for i in range(n):
        z = zcat[offsets[i]:offsets[i] + lengths[i]]
        ll += _country_ll_nb(z, rho, mu, sig, P, pi0)
    return ll


def _warmup_numba():
    global _NUMBA_WARMED
    if _NUMBA_WARMED or not HAS_NUMBA:
        return
    k = 3
    z = np.zeros(8, dtype=np.float64)
    rho = np.full(k, 0.5)
    mu = np.array([-0.1, 0.0, 0.1])
    sig = np.full(k, 0.05)
    P = np.full((k, k), 0.1)
    np.fill_diagonal(P, 0.8)
    pi0 = np.full(k, 1.0 / k)
    _country_ll_nb(z, rho, mu, sig, P, pi0)
    _panel_ll_nb(z, np.array([8], dtype=np.int64), np.array([0], dtype=np.int64),
                 rho, mu, sig, P, pi0)
    _NUMBA_WARMED = True


def _init_msar_worker():
    _warmup_numba()


def _run_one_start(payload):
    """Module-level worker so ProcessPoolExecutor can pickle it (Windows spawn)."""
    i, model, th0, packed, bounds, maxiter = payload
    opt = minimize(
        model._nll,
        np.asarray(th0, dtype=float),
        args=(packed,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-8},
    )
    return (
        int(i),
        float(opt.fun),
        np.asarray(opt.x, dtype=float).copy(),
        bool(opt.success),
        str(opt.message),
    )


def _country_loglik(z, rho, mu, sig, P, pi0, return_filter=False):
    """Hamilton filter. s' is drawn first; z' uses s'. rho, mu, sig are length-k."""
    z = np.ascontiguousarray(z, dtype=np.float64)
    rho = np.ascontiguousarray(rho, dtype=np.float64)
    mu = np.ascontiguousarray(mu, dtype=np.float64)
    sig = np.ascontiguousarray(sig, dtype=np.float64)
    P = np.ascontiguousarray(P, dtype=np.float64)
    pi0 = np.ascontiguousarray(pi0, dtype=np.float64)
    T = z.shape[0]
    k = mu.shape[0]
    if T < 2:
        return -1e10, None
    if not return_filter:
        return _country_ll_nb(z, rho, mu, sig, P, pi0), None

    logP = np.log(np.clip(P, 1e-12, 1.0))
    log_sig = np.log(sig)
    var0 = sig ** 2 / np.maximum(1.0 - rho ** 2, 1e-8)
    sd0 = np.sqrt(var0)
    log_f0 = -0.5 * LOG2PI - np.log(sd0) - 0.5 * ((z[0] - mu) / sd0) ** 2
    log_joint = np.log(np.clip(pi0, 1e-12, 1.0)) + log_f0
    log_p = logsumexp(log_joint)
    ll = log_p
    log_filt = log_joint - log_p
    filtered = np.empty((T, k))
    filtered[0] = np.exp(log_filt)
    one_m_rho = 1.0 - rho
    for t in range(T - 1):
        log_pred = logsumexp(log_filt[:, None] + logP, axis=0)
        mean = mu * one_m_rho + rho * z[t]
        log_f = -0.5 * LOG2PI - log_sig - 0.5 * ((z[t + 1] - mean) / sig) ** 2
        log_num = log_pred + log_f
        log_p = logsumexp(log_num)
        ll += log_p
        log_filt = log_num - log_p
        filtered[t + 1] = np.exp(log_filt)
    return ll, filtered


@dataclass
class PanelMSARResults:
    success: bool
    message: str
    nobs: int
    n_countries: int
    n_regimes: int
    loglik: float
    params: dict
    theta: np.ndarray
    param_names: list
    stderr: Optional[np.ndarray]
    country_ids: list
    filtered_probs: dict = field(default_factory=dict)
    time_base: float = 0.0
    time_step: float = 1.0
    se_params: Optional[dict] = None
    dropped_countries: list = field(default_factory=list)
    n_input_countries: int = 0
    n_input_rows: int = 0
    n_trimmed_spells: int = 0
    warnings: list = field(default_factory=list)
    has_numba: bool = HAS_NUMBA
    common_rho: bool = True
    rho_max: float = 0.995

    def summary(self) -> str:
        k = self.n_regimes
        pr = self.params
        se = self.se_params
        have_se = se is not None
        W = 12
        lab = 16
        mu = np.atleast_1d(pr["mu"]).astype(float)
        sig = np.atleast_1d(pr["sigma"]).astype(float)
        rho = np.atleast_1d(pr["rho"]).astype(float)
        P = np.asarray(pr["P"], dtype=float)
        se_mu = np.atleast_1d(se["mu"]) if have_se else None
        se_sig = np.atleast_1d(se["sigma"]) if have_se else None
        se_rho = np.atleast_1d(se["rho"]) if have_se and se is not None and "rho" in se else None
        se_P = np.asarray(se["P"]) if have_se and "P" in se else None

        def se_at(arr, i):
            if arr is None or arr.size <= i:
                return None
            return float(arr[i])

        def header_regimes(prefix=""):
            return f"{prefix:<{lab}}" + "".join(f"{s:{W}d}" for s in range(k))

        def est_row(name, vals):
            body = "".join(_cell_est(vals[s], W) for s in range(len(vals)))
            return f"{name:<{lab}}" + body

        def se_row(vals_se):
            cells = []
            for s in range(k):
                cells.append(_cell_se(se_at(vals_se, s), W))
            return f"{'':<{lab}}" + "".join(cells)

        lines = [
            "Two-step panel MS-AR(1): pooled OLS trend, then MS-AR on residuals",
            (
                f"Regimes: {k}    Countries: {self.n_countries}    "
                f"Observations: {self.nobs}"
            ),
            f"Log-likelihood (cycle): {self.loglik:.4f}",
            f"Converged: {self.success}    {self.message}",
            "",
            (
                "Pooled OLS removes a common intercept and a common slope "
                "before the likelihood. Those coefficients have no Hessian "
                "standard errors. a is the intercept at the first sample date."
            ),
            f"{'a (OLS)':<{lab}}{_cell_est(float(pr['a']), W)}",
            f"{'g (OLS)':<{lab}}{_cell_est(float(pr['g']), W)}",
            "",
            (
                "Unconditional E[z] restricted to 0. "
                "Regimes ordered by mu; one regime mean is implied."
            ),
        ]
        if self.common_rho:
            r = float(rho[0])
            lines.append(f"{'rho (common)':<{lab}}{_cell_est(r, W)}")
            if have_se and se_rho is not None:
                lines.append(f"{'':<{lab}}{_cell_se(se_at(se_rho, 0), W)}")
        lines.append("")
        lines.append("Regime parameters")
        lines.append(header_regimes())
        lines.append(est_row("mu", mu))
        if have_se:
            lines.append(se_row(se_mu))
        sig_row = sig if sig.size == k else np.full(k, float(sig[0]))
        lines.append(est_row("sigma", sig_row))
        if have_se:
            lines.append(se_row(se_sig))
        if not self.common_rho:
            lines.append(est_row("rho", rho))
            if have_se:
                lines.append(se_row(se_rho))
        lines.append("")
        lines.append("Transition matrix Pi [from \\ to]")
        lines.append(header_regimes())
        for i in range(k):
            lines.append(est_row(f"from {i}", P[i]))
            if have_se:
                if se_P is None:
                    lines.append(se_row(None))
                else:
                    lines.append(se_row(se_P[i]))
        if "pi" in pr:
            piv = np.atleast_1d(pr["pi"]).astype(float)
            lines.append(est_row("pi", piv))
            if have_se and se.get("pi") is not None:
                lines.append(se_row(np.atleast_1d(se["pi"])))
        if "Ez" in pr:
            ez = float(pr["Ez"])
            se_ez = se.get("Ez") if have_se else None
            lines.append(f"{'E[z] ergodic':<{lab}}{_cell_est(ez, W)}")
            if have_se:
                lines.append(f"{'':<{lab}}{_cell_se(se_ez, W)}")
        if self.dropped_countries:
            lines.append("")
            lines.append("Dropped countries:")
            for cid, reason in self.dropped_countries:
                lines.append(f"  {cid}: {reason}")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def plot_detrended(self, path, title=None):
        """Write a PDF of the pooled-OLS residuals."""
        if not self.filtered_probs:
            raise RuntimeError(
                "No stored cycles. Call fit(..., store_filtered=True) "
                "or pass detrend_pdf= to fit()."
            )
        import matplotlib.pyplot as plt

        ids = [cid for cid in self.country_ids if cid in self.filtered_probs]
        n = len(ids)
        cmap1 = plt.colormaps["tab20"]
        cmap2 = plt.colormaps["tab20b"]
        colors = [cmap1(i) for i in range(min(n, 20))]
        if n > 20:
            colors += [cmap2(i) for i in range(n - 20)]
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for i, cid in enumerate(ids):
            d = self.filtered_probs[cid]
            ax.plot(d["time"], d["cycle"], color=colors[i], lw=1.15, alpha=0.9, label=str(cid))
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("cycle (OLS residual)")
        if title is None:
            title = "Pooled OLS residual (common a and g removed)"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(
            ncol=2, fontsize=7, loc="center left",
            bbox_to_anchor=(1.01, 0.5), frameon=False, borderaxespad=0,
        )
        fig.tight_layout()
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return path


class PanelMSAR:
    """Pooled OLS detrending, then joint MLE of a panel MS-AR(1) on the residual.

    Parameters
    ----------
    n_regimes : int
        Integer >= 1. Regimes are ordered by mu after estimation. One mean
        is solved so the ergodic mean of z is 0.
    common_rho : bool
        If True (default), one AR(1) persistence for every regime.
        If False, each regime has its own rho.
    min_t : int
        Drop countries shorter than this many observations (after keeping
        the longest spell). Not years. Must be at least 2.
    rho_max : float
        Strict upper bound on |rho|. Unconstrained parameter is
        artanh(rho / rho_max). Default 0.995.
    """

    def __init__(self, n_regimes=3, common_rho=True, min_t=8, rho_max=RHO_MAX):
        if not isinstance(n_regimes, (int, np.integer)):
            raise TypeError(
                f"n_regimes must be an integer (got {type(n_regimes).__name__})."
            )
        n_regimes = int(n_regimes)
        if n_regimes < 1:
            raise ValueError(
                f"n_regimes must be a positive integer (got {n_regimes})."
            )
        if not isinstance(min_t, (int, np.integer)):
            raise TypeError(f"min_t must be an integer (got {type(min_t).__name__}).")
        if int(min_t) < 2:
            raise ValueError(
                f"min_t must be at least 2 because the AR(1) likelihood "
                f"needs two observations (got {min_t})."
            )
        self.n_regimes = int(n_regimes)
        self.common_rho = bool(common_rho)
        self.min_t = int(min_t)
        rho_max = float(rho_max)
        if not np.isfinite(rho_max) or rho_max <= 0.0:
            raise ValueError(
                f"rho_max must be positive and finite (got {rho_max})."
            )
        self.rho_max = rho_max
        self.res_ = None
        self._ols = None

    def _prepare(self, country, time, y):
        country = _as_1d("country", country)
        time = _coerce_time("time", time)
        y = _as_numeric_1d("y", y)
        n = country.shape[0]
        if time.shape[0] != n or y.shape[0] != n:
            raise ValueError(
                "country, time, and y must have the same length "
                f"(got {n}, {time.shape[0]}, {y.shape[0]}). "
                "Pass aligned arrays (index labels are ignored)."
            )
        if n == 0:
            raise ValueError("Empty input: country/time/y have length 0.")

        n_bad_y = int(np.sum(~np.isfinite(y)))
        n_bad_t = int(np.sum(~np.isfinite(time)))
        keep = np.isfinite(y) & np.isfinite(time) & pd.notna(country)
        n_drop_rows = int((~keep).sum())
        if keep.sum() == 0:
            raise ValueError(
                "Every row has a missing/non-finite y or time, or a missing "
                "country id. Check the input columns."
            )

        df = pd.DataFrame(
            {"country": country[keep], "time": time[keep], "y": y[keep]}
        )
        df = df.sort_values(["country", "time"])
        n_input_countries = int(pd.Series(country[pd.notna(country)]).nunique())

        panels, ids = [], []
        dropped = []
        n_trimmed = 0
        steps = []
        warnings = []
        year_dot_quarter = False

        if n_drop_rows:
            bits = []
            if n_bad_y:
                bits.append(f"{n_bad_y} non-finite y")
            if n_bad_t:
                bits.append(f"{n_bad_t} non-finite time")
            n_bad_c = int(np.sum(pd.isna(country)))
            if n_bad_c:
                bits.append(f"{n_bad_c} missing country")
            warnings.append(
                f"Dropped {n_drop_rows} row(s) with missing/non-finite values"
                + (f" ({', '.join(bits)})." if bits else ".")
            )

        y_keep = df["y"].to_numpy()
        if np.nanmedian(np.abs(y_keep)) > 20 or np.nanmax(np.abs(y_keep)) > 50:
            warnings.append(
                "y looks large for a log series "
                f"(median |y|={np.nanmedian(np.abs(y_keep)):.3g}, "
                f"max |y|={np.nanmax(np.abs(y_keep)):.3g}). "
                "The model is specified for log y, not levels."
            )

        for cid, g in df.groupby("country", sort=True):
            g = g.sort_values("time")
            t = g["time"].to_numpy()
            dt = np.diff(t)
            if dt.size:
                dmin, dmax = float(dt.min()), float(dt.max())
                if np.isclose(dmin, 0.1, atol=0.03) and 0.5 <= dmax <= 0.9:
                    year_dot_quarter = True
            if dt.size and np.any(dt <= 0):
                n_dup = int(np.sum(dt == 0))
                raise ValueError(
                    f"Duplicate time values in country {cid!r} "
                    f"({n_dup} duplicate step(s)). "
                    "Each country-time pair must be unique."
                )
            if len(g) < self.min_t:
                dropped.append(
                    (cid, f"only {len(g)} observations (min_t={self.min_t})")
                )
                continue
            if dt.size and not np.allclose(dt, dt[0]):
                step = np.median(dt)
                cuts = np.where(dt > step * 1.01)[0] + 1
                parts = np.split(np.arange(len(g)), cuts)
                idx = max(parts, key=len)
                if len(idx) < self.min_t:
                    dropped.append(
                        (
                            cid,
                            f"longest contiguous spell has {len(idx)} "
                            f"observations (min_t={self.min_t}); gaps in "
                            f"calendar time were split rather than interpolated",
                        )
                    )
                    continue
                n_trimmed += 1
                g = g.iloc[idx]
                t = g["time"].to_numpy()
                dt = np.diff(t)
            if dt.size:
                steps.append(float(dt[0]))
            panels.append((g["y"].to_numpy(), t))
            ids.append(cid)

        if not panels:
            n_in = n_input_countries
            raise ValueError(
                "No countries left after min_t / gap filters. "
                f"Started with {n_in} country id(s), min_t={self.min_t}. "
                "Either lower min_t, fill calendar gaps, or pass a longer panel. "
                "Internal missing periods are not interpolated; the longest "
                "contiguous spell is kept."
            )

        if year_dot_quarter:
            warnings.append(
                "Time values look like year.quarter codes "
                "(1970.1, 1970.2, 1970.3, 1970.4) rather than a linear "
                "scale. The Q4→Q1 wrap then looks like a gap and spells are "
                "split. Pass year + (quarter-1)/4 "
                "(1970.0, 1970.25, 1970.5, 1970.75) or an integer period "
                "index that increases by 1 each quarter."
            )

        first_t = np.array([t[0] for _, t in panels], dtype=float)
        if len(panels) > 1 and np.allclose(first_t, 0.0):
            warnings.append(
                "Every country starts at time 0. If `time` is periods-since-entry "
                "rather than a common calendar origin, the common trend is "
                "misspecified. Pass calendar time on one scale for every country "
                "(e.g. 1970.0, 1970.25, … for quarterly in years, or a period "
                "index with a shared origin)."
            )

        time_step = 1.0
        if steps:
            step0 = steps[0]
            if not np.allclose(steps, step0):
                time_step = float(np.median(steps))
                warnings.append(
                    "Countries do not share a common time increment "
                    f"(range {min(steps):g} to {max(steps):g}). "
                    "The AR(1) treats consecutive rows as lag-1 regardless of "
                    "the calendar gap; g is per unit of `time`."
                )
            else:
                time_step = float(step0)

        t0 = min(t.min() for _, t in panels)
        panels = [(yy, tt - t0) for yy, tt in panels]
        info = {
            "dropped": dropped,
            "n_input_countries": n_input_countries,
            "n_input_rows": n,
            "n_trimmed_spells": n_trimmed,
            "warnings": warnings,
            "time_base": float(t0),
            "time_step": float(time_step),
        }
        return panels, ids, float(t0), info

    def _n_trans(self):
        k = self.n_regimes
        return k * (k - 1)

    def _ez_pin_index(self):
        """Regime mean solved so that E[z] = 0. Not a free parameter."""
        return self.n_regimes - 1

    def _free_mu_indices(self):
        pin = self._ez_pin_index()
        return [s for s in range(self.n_regimes) if s != pin]

    def _mu_with_zero_ez(self, mu, rho, P):
        """Fill the pinned regime mean so the ergodic mean of z is 0."""
        k = self.n_regimes
        pin = self._ez_pin_index()
        w = np.zeros(k)
        _, ez0 = ergodic_cycle_mean(np.zeros(k), rho, P)
        for s in range(k):
            basis = np.zeros(k)
            basis[s] = 1.0
            _, ez = ergodic_cycle_mean(basis, rho, P)
            w[s] = ez - ez0
        mu = np.array(mu, dtype=float, copy=True)
        if abs(w[pin]) < 1e-8:
            mu[pin] = 0.0
            return mu
        acc = ez0
        for s in self._free_mu_indices():
            acc += w[s] * float(mu[s])
        mu[pin] = -acc / w[pin]
        return mu

    def param_names(self):
        k = self.n_regimes
        names = []
        for i in range(k):
            for j in range(k):
                if j == k - 1:
                    continue
                names.append(f"logitP[{i}->{j}]")
        if not self.common_rho:
            names += [f"rho[{s}]" for s in range(k)]
        else:
            names += ["rho"]
        names += [f"mu[{s}]" for s in self._free_mu_indices()]
        names += [f"sigma[{s}]" for s in range(k)]
        return names

    def _unpack(self, theta):
        k = self.n_regimes
        theta = np.asarray(theta, dtype=float)
        expected = len(self.param_names())
        if theta.ndim != 1 or theta.size != expected:
            raise ValueError(
                f"Internal parameter vector has length {theta.size}, "
                f"expected {expected}."
            )
        i = 0
        raw = theta[i:i + self._n_trans()].reshape(k, k - 1)
        i += self._n_trans()
        logits = np.zeros((k, k))
        logits[:, : k - 1] = raw
        P = _softmax_rows(logits)

        if not self.common_rho:
            rho = _rho_from_u(theta[i:i + k], self.rho_max)
            i += k
        else:
            rho = np.full(k, float(_rho_from_u(theta[i], self.rho_max)))
            i += 1

        mu = np.zeros(k)
        for s in self._free_mu_indices():
            mu[s] = theta[i]
            i += 1
        mu = self._mu_with_zero_ez(mu, rho, P)

        sig = np.exp(np.clip(theta[i:i + k], -20.0, 5.0))
        return {"P": P, "rho": rho, "mu": mu, "sigma": sig}

    def _pack_from_dicts(self, P, rho, mu, sig):
        k = self.n_regimes
        logits = np.log(np.clip(P, 1e-12, 1.0))
        raw = logits[:, : k - 1] - logits[:, k - 1][:, None]
        th = list(raw.ravel())
        if not self.common_rho:
            th += [float(_u_from_rho(r, self.rho_max)) for r in rho]
        else:
            th += [float(_u_from_rho(np.mean(rho), self.rho_max))]
        th += [float(mu[s]) for s in self._free_mu_indices()]
        th += [float(np.log(max(float(s), 1e-12))) for s in sig]
        return np.asarray(th, dtype=float)

    def _stack_panels(self, panels):
        lengths = np.array([len(y) for y, _ in panels], dtype=np.int64)
        offsets = np.zeros(len(panels), dtype=np.int64)
        offsets[1:] = np.cumsum(lengths[:-1])
        ycat = np.concatenate([y for y, _ in panels]).astype(np.float64)
        tcat = np.concatenate([t for _, t in panels]).astype(np.float64)
        return ycat, tcat, lengths, offsets

    @staticmethod
    def _ols_ag(y, t):
        X = np.column_stack([np.ones(len(y)), t])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(beta[0]), float(beta[1])

    def _detrend_panels(self, panels):
        """Pooled OLS of y on 1 and t. Return residual panels and (a, g)."""
        ycat, tcat, _, _ = self._stack_panels(panels)
        a, g = self._ols_ag(ycat, tcat)
        out = []
        for y, t in panels:
            e = np.asarray(y, dtype=float) - a - g * np.asarray(t, dtype=float)
            out.append((e, np.asarray(t, dtype=float)))
        return out, a, g

    def _nll(self, theta, packed):
        zcat, _tcat, lengths, offsets = packed
        p = self._unpack(theta)
        pi0 = _stationary_probs(p["P"]).astype(np.float64)
        ll = _panel_ll_nb(
            np.ascontiguousarray(zcat, dtype=np.float64),
            lengths,
            offsets,
            np.ascontiguousarray(p["rho"], dtype=np.float64),
            np.ascontiguousarray(p["mu"], dtype=np.float64),
            np.ascontiguousarray(p["sigma"], dtype=np.float64),
            np.ascontiguousarray(p["P"], dtype=np.float64),
            pi0,
        )
        if not np.isfinite(ll):
            return 1e12
        return -float(ll)

    def _starting_values(self, panels, n_starts, rng):
        resid = np.concatenate([y for y, _ in panels])
        s_hat = float(np.std(resid, ddof=1)) or 0.05

        rhos = []
        for y, _t in panels:
            z = np.asarray(y, dtype=float)
            if z.size < 4:
                continue
            zc = z - z.mean()
            den = float(np.dot(zc[:-1], zc[:-1]))
            if den <= 0:
                continue
            rhos.append(float(np.dot(zc[1:], zc[:-1]) / den))
        if rhos:
            rho0 = float(np.clip(np.median(rhos), 0.2, min(0.95, self.rho_max - 1e-4)))
        else:
            rho0 = 0.7

        k = self.n_regimes
        mid = k // 2
        if k == 1:
            P0 = np.ones((1, 1))
        else:
            sticky = 0.88
            off = (1.0 - sticky) / (k - 1)
            P0 = np.full((k, k), off)
            np.fill_diagonal(P0, sticky)

        def spread_mu(spread):
            mu = np.zeros(k)
            spread = abs(float(spread))
            if k == 1:
                return mu
            for j in range(1, mid + 1):
                if mid - j >= 0:
                    mu[mid - j] = -j * spread
                if mid + j < k:
                    mu[mid + j] = j * spread
            return mu

        def one(rho, mu_spread, sigs, P, jitter=0.0):
            mu = spread_mu(mu_spread)
            th = self._pack_from_dicts(P, np.full(k, rho), mu, np.asarray(sigs, float))
            if jitter:
                th = th + rng.normal(0.0, jitter, size=th.shape)
            return th

        dist = np.abs(np.arange(k) - mid).astype(float)
        scale = dist / max(mid, 1)
        sig_het = s_hat * (0.8 + 0.4 * scale)
        sig_base = np.full(k, s_hat)
        rho_hi = float(min(0.95, self.rho_max - 1e-4))
        templates = [
            (rho0, s_hat, sig_het),
            (0.5, 0.5 * s_hat, sig_base),
            (min(0.85, rho_hi), 1.5 * s_hat, sig_het),
            (0.6, s_hat, sig_base),
            (rho_hi, s_hat, sig_het),
            (rho0, 2.0 * s_hat, sig_het),
            (0.3, s_hat, sig_base),
        ]
        starts = []
        for rho, spread, sigs in templates:
            starts.append(one(rho, spread, sigs, P0))
        while len(starts) < n_starts:
            starts.append(
                one(
                    float(rng.uniform(0.3, min(0.9, self.rho_max - 1e-4))),
                    abs(float(rng.normal(0, s_hat))),
                    np.maximum(sig_het * rng.uniform(0.7, 1.4, size=k), 1e-4),
                    P0,
                    jitter=0.08,
                )
            )
        return starts[:n_starts]

    def _order_regimes(self, theta):
        """Permute so mu is increasing. E[z]=0 is reimposed by unpack."""
        p = self._unpack(theta)
        order = np.argsort(p["mu"], kind="mergesort")
        mu = p["mu"][order]
        rho = p["rho"][order]
        sig = p["sigma"][order]
        P = p["P"][np.ix_(order, order)]
        return self._pack_from_dicts(P, rho, mu, sig)

    def _se_transformed(self, theta, se_raw, cov=None):
        """Delta-method SEs on the model parameterization (not logits)."""
        if se_raw is None:
            return None
        se_raw = np.asarray(se_raw, dtype=float)
        names = self.param_names()
        raw = {n: float(s) for n, s in zip(names, se_raw)}
        p = self._unpack(theta)
        k = self.n_regimes
        out = {}

        cap = float(self.rho_max)
        if not self.common_rho:
            out["rho"] = np.array([
                raw[f"rho[{s}]"] * cap * (1.0 - (p["rho"][s] / cap) ** 2)
                for s in range(k)
            ])
        else:
            r = float(p["rho"][0])
            out["rho"] = raw["rho"] * cap * (1.0 - (r / cap) ** 2)

        if cov is not None:
            out["mu"] = self._mu_se_zero_ez(theta, cov)
        else:
            mu_se = np.full(k, np.nan)
            for s in self._free_mu_indices():
                mu_se[s] = raw[f"mu[{s}]"]
            out["mu"] = mu_se

        out["sigma"] = np.array([
            raw[f"sigma[{s}]"] * float(p["sigma"][s]) for s in range(k)
        ])
        out["P"] = self._se_P(p["P"], cov)
        return out

    def _mu_se_zero_ez(self, theta, cov):
        """Delta-method SEs for regime means when one mean enforces E[z]=0."""
        k = self.n_regimes
        theta = np.asarray(theta, dtype=float)
        cov = np.asarray(cov, dtype=float)
        base = self._unpack(theta)["mu"]
        eps = 1e-5
        jac = np.zeros((k, theta.size))
        for j in range(theta.size):
            th = theta.copy()
            th[j] += eps
            jac[:, j] = (self._unpack(th)["mu"] - base) / eps
        var = jac @ cov @ jac.T
        se = np.sqrt(np.clip(np.diag(var), 0.0, np.inf))
        se[~np.isfinite(se)] = np.nan
        return se

    def _se_P(self, P, cov):
        """Delta-method SEs for row-stochastic P from free logits."""
        k = P.shape[0]
        seP = np.full((k, k), np.nan)
        if cov is None or k <= 1:
            if k == 1:
                seP[0, 0] = 0.0
            return seP
        nfree = k - 1
        for i in range(k):
            sl = slice(i * nfree, (i + 1) * nfree)
            C = np.asarray(cov[sl, sl], dtype=float)
            p = P[i]
            for j in range(k):
                g = np.empty(nfree)
                for m in range(nfree):
                    if j == k - 1:
                        g[m] = -p[k - 1] * p[m]
                    else:
                        g[m] = p[j] * ((1.0 if j == m else 0.0) - p[m])
                var = float(g @ C @ g)
                seP[i, j] = np.sqrt(var) if np.isfinite(var) and var > 0 else np.nan
        return seP

    def fit(
        self,
        country,
        time,
        y,
        n_starts=7,
        maxiter=400,
        seed=1,
        compute_se=True,
        store_filtered=True,
        verbose=False,
        detrend_pdf=None,
    ):
        if not isinstance(n_starts, (int, np.integer)) or int(n_starts) < 1:
            raise ValueError(f"n_starts must be a positive integer (got {n_starts!r}).")
        if not isinstance(maxiter, (int, np.integer)) or int(maxiter) < 1:
            raise ValueError(f"maxiter must be a positive integer (got {maxiter!r}).")
        n_starts = int(n_starts)
        maxiter = int(maxiter)

        if not HAS_NUMBA:
            msg = (
                "numba is not installed. Joint MLE will run in pure Python and "
                "is typically ~50x too slow for multi-start estimation. "
                "Install numba before fitting real panels."
            )
            if verbose:
                print(msg)

        panels, ids, t0, info = self._prepare(country, time, y)
        warnings = list(info["warnings"])
        if not HAS_NUMBA:
            warnings.insert(0, (
                "numba is not installed; likelihood evaluations use pure Python."
            ))
        if self.n_regimes == 1:
            warnings.append(
                "n_regimes=1 is a non-switching AR(1). "
                "E[z]=0 forces the single mean to 0."
            )

        level_panels = panels
        panels, a_ols, g_ols = self._detrend_panels(level_panels)
        self._ols = (a_ols, g_ols)
        ycat, tcat, lengths, offsets = self._stack_panels(panels)
        if abs(float(ycat.sum())) > 1e-6 or abs(float(np.dot(ycat, tcat))) > 1e-6:
            warnings.append(
                "Pooled OLS residuals are not orthogonal to (1, t) "
                "at the expected numerical tolerance."
            )

        nobs = int(sum(len(yy) for yy, _ in panels))
        n_par = len(self.param_names())
        if nobs < 10 * n_par:
            warnings.append(
                f"Short panel relative to parameter count "
                f"({nobs} observations, {n_par} free parameters). "
                "Estimates may be imprecise."
            )
        if len(panels) < 2:
            warnings.append(
                "Only one country survived the sample filters. The estimator "
                "still runs, but this is no longer a panel."
            )

        rng = np.random.default_rng(seed)
        packed = (ycat, tcat, lengths, offsets)
        _warmup_numba()
        starts = self._starting_values(panels, n_starts, rng)
        bounds = None
        n_workers = min(n_starts, int(os.cpu_count() or n_starts))

        def _run_start(i, th0):
            opt = minimize(
                self._nll,
                th0,
                args=(packed,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-8},
            )
            return (
                i,
                float(opt.fun),
                np.asarray(opt.x, dtype=float).copy(),
                bool(opt.success),
                str(opt.message),
            )

        def _announce(rec):
            if verbose:
                i, fun, _, ok, msg = rec
                print(
                    f"  start {i + 1}/{n_starts}: nll={fun:.4f}  "
                    f"success={ok}  {msg}",
                    flush=True,
                )
            return rec

        if n_starts == 1 or n_workers == 1:
            raw = [
                _announce(_run_start(i, th0))
                for i, th0 in enumerate(starts)
            ]
        else:
            raw = []
            payloads = [
                (i, self, th0, packed, bounds, maxiter)
                for i, th0 in enumerate(starts)
            ]
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_msar_worker,
            ) as pool:
                futs = [pool.submit(_run_one_start, p) for p in payloads]
                for fut in as_completed(futs):
                    raw.append(_announce(fut.result()))
            raw.sort(key=lambda r: r[0])

        best, best_fun, best_msg, best_ok = None, np.inf, "", False
        n_ok = 0
        for _i, fun, x, ok, msg in raw:
            if ok:
                n_ok += 1
            if fun < best_fun:
                best_fun = fun
                best = x
                best_msg = msg
                best_ok = ok

        if best is None:
            raise RuntimeError(
                "Optimization failed on every start (no finite likelihood). "
                "Check that y is in logs, time is calendar time "
                "on a common origin, and the panel is not constant."
            )

        best = self._order_regimes(best)
        opt = minimize(
            self._nll,
            best,
            args=(packed,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-9},
        )
        if opt.fun <= best_fun + 1e-6:
            best = self._order_regimes(opt.x)
            best_fun = float(opt.fun)
            best_msg = str(opt.message)
            best_ok = bool(opt.success)

        if not best_ok:
            warnings.append(
                f"Optimizer did not report success ({best_msg}). "
                "Estimates may still be usable; inspect the likelihood and "
                "try more starts."
            )
        if n_ok == 0:
            warnings.append("None of the multi-start runs reported success.")

        p = self._unpack(best)
        names = self.param_names()
        ll = -best_fun
        if compute_se:
            stderr, cov = self._stderr(best, packed)
            se_params = self._se_transformed(best, stderr, cov)
        else:
            stderr, se_params, cov = None, None, None
        if compute_se and stderr is not None and not np.any(np.isfinite(stderr)):
            warnings.append(
                "Numerical Hessian could not be inverted; std errors are missing. "
                "For a paper, bootstrap countries."
            )

        mu = np.asarray(p["mu"])
        k = self.n_regimes
        if k >= 2:
            gaps = np.diff(np.sort(mu))
            if np.any(gaps < 1e-3):
                warnings.append(
                    "At least two regime means collapsed (gap < 1e-3). "
                    "Coefficient estimates for those regimes may not be "
                    "separately identified."
                )
        sig = np.asarray(p["sigma"])
        if np.max(sig) > 10 * max(np.min(sig), 1e-8):
            warnings.append(
                "Regime standard deviations differ by more than 10x; "
                "one sigma may have exploded."
            )

        if detrend_pdf:
            store_filtered = True
        filtered = {}
        if store_filtered:
            pi0 = _stationary_probs(p["P"])
            for cid, (zz, tt) in zip(ids, panels):
                _, filt = _country_loglik(
                    zz, p["rho"], p["mu"], p["sigma"], p["P"], pi0, return_filter=True
                )
                out = {"time": tt + t0, "cycle": zz}
                for s in range(self.n_regimes):
                    out[f"p_regime{s}"] = filt[:, s]
                filtered[cid] = pd.DataFrame(out)

        rho_out = float(p["rho"][0]) if self.common_rho else p["rho"]
        pi_hat, ez_hat = ergodic_cycle_mean(p["mu"], p["rho"], p["P"])
        params = {
            "P": p["P"],
            "rho": rho_out,
            "mu": p["mu"],
            "sigma": p["sigma"],
            "g": float(g_ols),
            "a": float(a_ols),
            "pi": pi_hat,
            "Ez": ez_hat,
        }
        if se_params is not None and cov is not None:
            def _pi_ez(th):
                pp = self._unpack(th)
                pi2, ez2 = ergodic_cycle_mean(pp["mu"], pp["rho"], pp["P"])
                return np.concatenate([np.asarray(pi2, dtype=float), [ez2]])
            try:
                se_pez = self._delta_se(best, cov, _pi_ez)
                se_params["pi"] = se_pez[:-1]
                se_params["Ez"] = float(se_pez[-1])
            except Exception:
                pass

        self.res_ = PanelMSARResults(
            success=best_ok,
            message=best_msg,
            nobs=nobs,
            n_countries=len(panels),
            n_regimes=self.n_regimes,
            loglik=ll,
            params=params,
            theta=best,
            param_names=names,
            stderr=stderr,
            country_ids=ids,
            filtered_probs=filtered,
            time_base=t0,
            time_step=info["time_step"],
            se_params=se_params,
            dropped_countries=info["dropped"],
            n_input_countries=info["n_input_countries"],
            n_input_rows=info["n_input_rows"],
            n_trimmed_spells=info["n_trimmed_spells"],
            warnings=warnings,
            has_numba=HAS_NUMBA,
            common_rho=self.common_rho,
            rho_max=self.rho_max,
        )
        if detrend_pdf:
            self.res_.plot_detrended(detrend_pdf)
        return self.res_

    def _stderr(self, theta, packed):
        theta = np.asarray(theta, dtype=float)
        n = theta.size
        eps = 1e-4 * (1.0 + np.abs(theta))
        f0 = self._nll(theta, packed)
        H = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = eps[i]
            for j in range(i, n):
                ej = np.zeros(n)
                ej[j] = eps[j]
                if i == j:
                    fp = self._nll(theta + ei, packed)
                    fm = self._nll(theta - ei, packed)
                    H[i, i] = (fp - 2.0 * f0 + fm) / (eps[i] ** 2)
                else:
                    fpp = self._nll(theta + ei + ej, packed)
                    fpm = self._nll(theta + ei - ej, packed)
                    fmp = self._nll(theta - ei + ej, packed)
                    fmm = self._nll(theta - ei - ej, packed)
                    val = (fpp - fpm - fmp + fmm) / (4.0 * eps[i] * eps[j])
                    H[i, j] = H[j, i] = val
        se = np.full(n, np.nan)
        cov = None
        try:
            cov = np.linalg.inv(H)
            diag = np.diag(cov)
            se = np.sqrt(np.maximum(diag, 0.0))
            se[~np.isfinite(diag) | (diag <= 0)] = np.nan
        except np.linalg.LinAlgError:
            cov = None
        return se, cov

    def _delta_se(self, theta, cov, fun, eps=1e-5):
        """Delta-method SE of a vector function of unconstrained theta."""
        theta = np.asarray(theta, dtype=float)
        f0 = np.atleast_1d(fun(theta)).astype(float)
        n = theta.size
        G = np.zeros((f0.size, n))
        step = eps * (1.0 + np.abs(theta))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = step[i]
            fp = np.atleast_1d(fun(theta + ei)).astype(float)
            fm = np.atleast_1d(fun(theta - ei)).astype(float)
            G[:, i] = (fp - fm) / (2.0 * step[i])
        V = G @ np.asarray(cov, dtype=float) @ G.T
        d = np.diag(V)
        se = np.sqrt(np.clip(d, 0.0, None))
        se[~np.isfinite(se) | (d <= 0)] = np.nan
        return se

    def bootstrap_se(
        self, country, time, y, theta, B=40, seed=11, maxiter=180, verbose=True,
    ):
        """Country-resampling bootstrap SEs. Each draw redoes the pooled OLS."""
        panels, _ids, _t0, _info = self._prepare(country, time, y)
        n_c = len(panels)
        rng = np.random.default_rng(seed)
        theta = np.asarray(theta, dtype=float)
        rec_mu, rec_sig, rec_rho, rec_P, rec_pi, rec_ez = (
            [], [], [], [], [], [],
        )
        n_ok = 0
        for b in range(int(B)):
            pick = rng.integers(0, n_c, size=n_c)
            boot, _a, _g = self._detrend_panels([panels[i] for i in pick])
            packed = self._stack_panels(boot)
            opt = minimize(
                self._nll,
                theta,
                args=(packed,),
                method="L-BFGS-B",
                options={"maxiter": int(maxiter), "ftol": 1e-8},
            )
            if (not np.isfinite(opt.fun)) or opt.fun > 1e8:
                if verbose:
                    print(f"  boot {b + 1}/{B}: skip (nll={opt.fun})", flush=True)
                continue
            th = self._order_regimes(opt.x)
            p = self._unpack(th)
            pi, ez = ergodic_cycle_mean(p["mu"], p["rho"], p["P"])
            rec_mu.append(np.asarray(p["mu"], dtype=float))
            rec_sig.append(np.asarray(p["sigma"], dtype=float))
            rec_rho.append(np.asarray(p["rho"], dtype=float))
            rec_P.append(np.asarray(p["P"], dtype=float))
            rec_pi.append(pi)
            rec_ez.append(ez)
            n_ok += 1
            if verbose:
                print(
                    f"  boot {b + 1}/{B}: nll={opt.fun:.4f}  Ez={ez:.4f}",
                    flush=True,
                )
        if n_ok < 8:
            return None, n_ok
        se = {
            "mu": np.std(np.vstack(rec_mu), axis=0, ddof=1),
            "sigma": np.std(np.vstack(rec_sig), axis=0, ddof=1),
            "rho": np.std(np.vstack(rec_rho), axis=0, ddof=1),
            "P": np.std(np.stack(rec_P, axis=0), axis=0, ddof=1),
            "pi": np.std(np.vstack(rec_pi), axis=0, ddof=1),
            "Ez": float(np.std(np.asarray(rec_ez), ddof=1)),
        }
        return se, n_ok


def simulate_panel(
    n_countries=40,
    t_min=28,
    t_max=50,
    year0=1970,
    time_step=1.0,
    g=0.018,
    a=1.0,
    rho=0.75,
    mu=(-0.20, 0.0, 0.20),
    sigma=(0.06, 0.035, 0.06),
    stay=(0.90, 0.88, 0.90),
    seed=7,
    unbalanced=True,
):
    """Draw an unbalanced panel from a common trend plus an MS-AR(1).

    time_step is the calendar spacing of consecutive observations (1 = annual
    if year0 is a calendar year; 0.25 = quarterly on a yearly scale). rho is
    per observation; g is per unit of calendar time. y = a + g * years-since-
    year0 + z, and the returned time column is the calendar date.
    """
    if int(n_countries) < 1:
        raise ValueError(f"n_countries must be >= 1 (got {n_countries}).")
    if int(t_min) < 2 or int(t_max) < int(t_min):
        raise ValueError(
            f"Need 2 <= t_min <= t_max (got t_min={t_min}, t_max={t_max})."
        )
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    stay = np.asarray(stay, dtype=float)
    if mu.ndim != 1 or sigma.shape != mu.shape or stay.shape != mu.shape:
        raise ValueError(
            "mu, sigma, and stay must be 1-d arrays of the same length "
            f"(got {mu.shape}, {sigma.shape}, {stay.shape})."
        )
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive.")
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError(f"rho must satisfy |rho| < 1 (got {rho}).")
    if not np.isfinite(time_step) or float(time_step) <= 0:
        raise ValueError(f"time_step must be positive (got {time_step}).")
    time_step = float(time_step)

    rng = np.random.default_rng(seed)
    k = mu.size
    rho_v = np.full(k, rho, dtype=float)
    if k == 1:
        P = np.ones((1, 1))
    else:
        if np.any(stay <= 0) or np.any(stay >= 1):
            raise ValueError("stay probabilities must lie in (0, 1).")
        P = np.empty((k, k))
        for s in range(k):
            off = (1.0 - stay[s]) / (k - 1)
            P[s] = off
            P[s, s] = stay[s]
    pi0 = _stationary_probs(P)

    rows = []
    for i in range(int(n_countries)):
        T = int(rng.integers(t_min, t_max + 1)) if unbalanced else int(t_max)
        start = int(rng.integers(0, 12)) if unbalanced else 0
        t = (start + np.arange(T)) * time_step
        cal = year0 + t
        s = int(rng.choice(k, p=pi0))
        sd0 = sigma[s] / np.sqrt(max(1.0 - rho_v[s] ** 2, 1e-8))
        z = rng.normal(mu[s], sd0)
        rows.append((i, cal[0], a + g * t[0] + z, s, z))
        for h in range(1, T):
            s = int(rng.choice(k, p=P[s]))
            z = mu[s] * (1.0 - rho_v[s]) + rho_v[s] * z + sigma[s] * rng.normal()
            rows.append((i, cal[h], a + g * t[h] + z, s, z))
    df = pd.DataFrame(rows, columns=["country", "time", "y", "s", "z"])
    df["year"] = df["time"]
    return df
