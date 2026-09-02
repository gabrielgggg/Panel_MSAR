"""
Joint panel Markov-switching AR(1) around a common log-linear trend.

    y_it = a + g * t + z_it

    z_{i,t+1} = (1 - rho(s_it)) * mu(s_it) + rho(s_it) * z_it
                + sigma(s_it) * eps_it

    s_{i,t+1} ~ Markov(P | s_it)

n_regimes must be odd so a unique median regime exists. After estimation,
regimes are ordered by mean and shifted so mu[k//2] = 0 (the shift is
absorbed into a or the a_i). Optional country intercepts a_i and/or
country slopes g_i are profiled out of the joint likelihood. Countries
are independent given shared parameters; latent regimes are
country-specific. The panel may be unbalanced.

Timing: the regime dated t governs the transition from z_t to z_{t+1};
then a new regime is drawn.

Time is numeric calendar time on a common origin, at whatever sampling
frequency the panel is observed (annual, quarterly, ...). Consecutive
rows are one AR(1) step: rho is per observation. g is per unit of the
time variable (per year if time is 1970.0, 1970.25, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
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


def _cell_se(se, width=12, pinned=False):
    if pinned:
        return f"{'(—)':>{width}}"
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
        for s in range(k):
            resid = (ztp - (mu[s] * one_m_rho[s] + rho[s] * zt)) / sig[s]
            log_num[s] = (
                log_filt[s] - 0.5 * LOG2PI - log_sig[s] - 0.5 * resid * resid
            )
        log_p = _logsumexp_nb(log_num)
        ll += log_p
        for s in range(k):
            log_num[s] -= log_p
        for sp in range(k):
            for s in range(k):
                acc[s] = log_num[s] + logP[s, sp]
            log_pred[sp] = _logsumexp_nb(acc)
        for s in range(k):
            log_filt[s] = log_pred[s]
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


def _country_loglik(z, rho, mu, sig, P, pi0, return_filter=False):
    """Hamilton filter, user's timing. rho, mu, sig are length-k."""
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
        mean = mu * one_m_rho + rho * z[t]
        log_f = -0.5 * LOG2PI - log_sig - 0.5 * ((z[t + 1] - mean) / sig) ** 2
        log_num = log_filt + log_f
        log_p = logsumexp(log_num)
        ll += log_p
        log_s_t = log_num - log_p
        log_filt = logsumexp(log_s_t[:, None] + logP, axis=0)
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
    common_sigma: bool = False
    country_intercepts: bool = False
    country_trends: bool = False

    def summary(self) -> str:
        k = self.n_regimes
        mid = k // 2
        pr = self.params
        se = self.se_params
        have_se = se is not None
        W = 12
        lab = 14
        mu = np.atleast_1d(pr["mu"]).astype(float)
        sig = np.atleast_1d(pr["sigma"]).astype(float)
        rho = np.atleast_1d(pr["rho"]).astype(float)
        P = np.asarray(pr["P"], dtype=float)
        se_mu = np.atleast_1d(se["mu"]) if have_se else None
        se_sig = np.atleast_1d(se["sigma"]) if have_se else None
        se_rho = np.atleast_1d(se["rho"]) if have_se else None
        se_P = np.asarray(se["P"]) if have_se and "P" in se else None

        def se_at(arr, i):
            if arr is None or arr.size <= i:
                return None
            return float(arr[i])

        def header_regimes(prefix=""):
            return f"{prefix:<{lab}}" + "".join(f"{s:{W}d}" for s in range(k))

        def est_row(name, vals, pinned=None):
            body = "".join(_cell_est(vals[s], W) for s in range(len(vals)))
            return f"{name:<{lab}}" + body

        def se_row(vals_se, pinned=None):
            cells = []
            for s in range(k):
                is_pin = bool(pinned[s]) if pinned is not None else False
                cells.append(_cell_se(se_at(vals_se, s), W, pinned=is_pin))
            return f"{'':<{lab}}" + "".join(cells)

        lines = [
            "Joint panel MS-AR(1) + common log-linear trend",
            (
                f"Regimes: {k}    Countries: {self.n_countries}    "
                f"Observations: {self.nobs}"
            ),
            f"Log-likelihood: {self.loglik:.4f}",
            f"Converged: {self.success}    {self.message}",
            "",
            f"Regimes ordered by mu; median regime {mid} has mu pinned at 0.",
        ]
        if self.country_intercepts:
            aa = np.array(list(pr["a"].values()) if isinstance(pr["a"], dict) else [pr["a"]], dtype=float)
            lines.append(
                f"{'a (country)':<{lab}}mean={aa.mean():.4f}  "
                f"min={aa.min():.4f}  max={aa.max():.4f}"
            )
        if self.country_trends:
            gg = np.array(list(pr["g"].values()) if isinstance(pr["g"], dict) else [pr["g"]], dtype=float)
            lines.append(
                f"{'g (country)':<{lab}}mean={gg.mean():.4f}  "
                f"min={gg.min():.4f}  max={gg.max():.4f}"
            )
        else:
            g = float(pr["g"])
            se_g = se.get("g") if have_se else None
            lines.append(f"{'g':<{lab}}{_cell_est(g, W)}")
            if have_se:
                lines.append(f"{'':<{lab}}{_cell_se(se_g, W)}")
        if self.common_rho:
            r = float(rho[0])
            lines.append(f"{'rho (common)':<{lab}}{_cell_est(r, W)}")
            if have_se:
                lines.append(f"{'':<{lab}}{_cell_se(se_at(se_rho, 0), W)}")
        if self.common_sigma:
            s0 = float(sig[0])
            lines.append(f"{'sigma (common)':<{lab}}{_cell_est(s0, W)}")
            if have_se:
                lines.append(f"{'':<{lab}}{_cell_se(se_at(se_sig, 0), W)}")
        lines.append("")
        lines.append("Regime parameters")
        lines.append(header_regimes())
        pin_mu = np.zeros(k, dtype=bool)
        pin_mu[mid] = True
        lines.append(est_row("mu", mu))
        if have_se:
            lines.append(se_row(se_mu, pinned=pin_mu))
        if not self.common_sigma:
            sig_row = sig if sig.size == k else np.full(k, float(sig[0]))
            lines.append(est_row("sigma", sig_row))
            if have_se:
                se_s = se_sig if se_sig is not None and se_sig.size == k else None
                lines.append(se_row(se_s if se_s is not None else np.full(k, se_at(se_sig, 0))))
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
        """Write a PDF of country cycles z_it = y_it - a_i - g_i t."""
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
        ax.set_ylabel("cycle (common or country trend removed)")
        if title is None:
            bits = []
            bits.append("a_i" if self.country_intercepts else "common a")
            bits.append("g_i" if self.country_trends else "common g")
            title = "Detrended series (" + ", ".join(bits) + ")"
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
    """Joint MLE: log-linear trend + panel MS-AR(1) cycle.

    Parameters
    ----------
    n_regimes : int
        Odd integer >= 1. The mean of the median regime (index k//2
        after ordering) is pinned at 0. Even k is rejected because it
        has no unique middle regime.
    common_rho : bool
        If True (default), one AR(1) persistence for every regime.
        If False, each regime has its own rho.
    common_sigma : bool
        If False (default), sigma switches with the regime.
        If True, one sigma for every regime.
    country_intercepts : bool
        If True, each country has its own intercept a_i (profiled).
        If False (default), one common a.
    country_trends : bool
        If True, each country has its own slope g_i (profiled).
        If False (default), one common g.
    min_t : int
        Drop countries shorter than this many *observations* (after keeping
        the longest spell). Not years — 8 is 8 quarters if the panel is
        quarterly. Must be at least 2.
    """

    def __init__(
        self,
        n_regimes=3,
        common_rho=True,
        common_sigma=False,
        country_intercepts=False,
        country_trends=False,
        min_t=8,
    ):
        if not isinstance(n_regimes, (int, np.integer)):
            raise TypeError(
                f"n_regimes must be an integer (got {type(n_regimes).__name__})."
            )
        n_regimes = int(n_regimes)
        if n_regimes < 1 or n_regimes % 2 == 0:
            raise ValueError(
                f"n_regimes must be a positive odd integer so a unique "
                f"median regime can have its mean pinned at 0 (got {n_regimes}). "
                "Even k (including 2) is not supported."
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
        self.common_sigma = bool(common_sigma)
        self.country_intercepts = bool(country_intercepts)
        self.country_trends = bool(country_trends)
        self.min_t = int(min_t)
        self.res_ = None
        self._ols = None
        self._ag_cache = None

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
        # Drop NA/inf rows; keep a copy of original country ids for reporting.
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

    def _mid_regime(self):
        """Index of the median regime after means are ordered."""
        return self.n_regimes // 2

    def _free_mu_indices(self):
        mid = self._mid_regime()
        return [s for s in range(self.n_regimes) if s != mid]

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
        if not self.common_sigma:
            names += [f"sigma[{s}]" for s in range(k)]
        else:
            names += ["sigma"]
        if not self.country_trends:
            names += ["g"]
        if not self.country_intercepts:
            names += ["a"]
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
            rho = np.tanh(theta[i:i + k])
            i += k
        else:
            rho = np.full(k, np.tanh(theta[i]))
            i += 1

        mu = np.zeros(k)
        for s in self._free_mu_indices():
            mu[s] = theta[i]
            i += 1

        if not self.common_sigma:
            sig = np.exp(np.clip(theta[i:i + k], -20.0, 5.0))
            i += k
        else:
            sig = np.full(k, np.exp(np.clip(theta[i], -20.0, 5.0)))
            i += 1

        if not self.country_trends:
            g = theta[i]
            i += 1
        else:
            g = 0.0
        if not self.country_intercepts:
            a = theta[i]
        else:
            a = 0.0
        return {"P": P, "rho": rho, "mu": mu, "sigma": sig, "g": g, "a": a}

    def _pack_from_dicts(self, P, rho, mu, sig, g, a):
        k = self.n_regimes
        logits = np.log(np.clip(P, 1e-12, 1.0))
        raw = logits[:, : k - 1] - logits[:, k - 1][:, None]
        th = list(raw.ravel())
        if not self.common_rho:
            th += [np.arctanh(np.clip(r, -0.99, 0.99)) for r in rho]
        else:
            th += [np.arctanh(np.clip(float(np.mean(rho)), -0.99, 0.99))]
        th += [float(mu[s]) for s in self._free_mu_indices()]
        if not self.common_sigma:
            th += [float(np.log(s)) for s in sig]
        else:
            th += [float(np.log(np.mean(sig)))]
        if not self.country_trends:
            th += [float(g)]
        if not self.country_intercepts:
            th += [float(a)]
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

    def _country_nll_ag(self, a, g, y, t, rho, mu, sig, P, pi0):
        z = y - a - g * t
        try:
            ll = _country_ll_nb(z, rho, mu, sig, P, pi0)
        except Exception:
            return 1e12
        if not np.isfinite(ll):
            return 1e12
        return -float(ll)

    def _brent_1d(self, fun, x0, half_width):
        lo, hi = float(x0) - half_width, float(x0) + half_width
        if lo > hi:
            lo, hi = hi, lo
        try:
            opt = minimize_scalar(
                fun, bounds=(lo, hi), method="bounded",
                options={"xatol": 1e-8, "maxiter": 40},
            )
            if np.isfinite(opt.fun):
                return float(opt.x)
        except Exception:
            pass
        return float(x0)

    def _newton_2d(self, y, t, rho, mu, sig, P, pi0, a0, g0):
        a, g = float(a0), float(g0)

        def f(aa, gg):
            return self._country_nll_ag(aa, gg, y, t, rho, mu, sig, P, pi0)

        f0 = f(a, g)
        for _ in range(8):
            ea = 1e-5 * (1.0 + abs(a))
            eg = 1e-6 * (1.0 + abs(g))
            f_ap, f_am = f(a + ea, g), f(a - ea, g)
            f_gp, f_gm = f(a, g + eg), f(a, g - eg)
            da = (f_ap - f_am) / (2.0 * ea)
            dg = (f_gp - f_gm) / (2.0 * eg)
            haa = (f_ap - 2.0 * f0 + f_am) / (ea ** 2)
            hgg = (f_gp - 2.0 * f0 + f_gm) / (eg ** 2)
            hag = (f(a + ea, g + eg) - f_ap - f_gp + f0) / (ea * eg)
            H = np.array([[haa, hag], [hag, hgg]], dtype=float)
            grad = np.array([da, dg], dtype=float)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)):
                break
            improved = False
            for lam in (1.0, 0.5, 0.25, 0.1):
                a2, g2 = a - lam * step[0], g - lam * step[1]
                f2 = f(a2, g2)
                if f2 < f0 - 1e-12:
                    a, g, f0 = a2, g2, f2
                    improved = True
                    break
            if (not improved) or (abs(step[0]) + abs(step[1]) < 1e-10):
                break
        return a, g, f0

    def _profile_one(self, y, t, p, pi0, a0, g0):
        """Max country ll over free a_i and/or g_i. Returns ll, a, g."""
        fix_a = None if self.country_intercepts else float(p["a"])
        fix_g = None if self.country_trends else float(p["g"])
        rho = np.ascontiguousarray(p["rho"], dtype=np.float64)
        mu = np.ascontiguousarray(p["mu"], dtype=np.float64)
        sig = np.ascontiguousarray(p["sigma"], dtype=np.float64)
        P = np.ascontiguousarray(p["P"], dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        t = np.ascontiguousarray(t, dtype=np.float64)

        if fix_a is not None and fix_g is not None:
            nll = self._country_nll_ag(fix_a, fix_g, y, t, rho, mu, sig, P, pi0)
            return -nll, fix_a, fix_g

        if fix_a is None and fix_g is not None:
            def fa(a):
                return self._country_nll_ag(a, fix_g, y, t, rho, mu, sig, P, pi0)
            a = self._brent_1d(fa, a0, half_width=5.0)
            return -fa(a), a, fix_g

        if fix_g is None and fix_a is not None:
            def fg(g):
                return self._country_nll_ag(fix_a, g, y, t, rho, mu, sig, P, pi0)
            g = self._brent_1d(fg, g0, half_width=0.15)
            return -fg(g), fix_a, g

        a, g, nll = self._newton_2d(y, t, rho, mu, sig, P, pi0, a0, g0)
        return -nll, a, g

    def _profile_all(self, theta, packed):
        ycat, tcat, lengths, offsets = packed
        p = self._unpack(theta)
        pi0 = _stationary_probs(p["P"]).astype(np.float64)
        n = int(lengths.shape[0])
        a_hat = np.empty(n)
        g_hat = np.empty(n)
        ll = 0.0
        starts = self._ag_cache if self._ag_cache is not None else self._ols
        for i in range(n):
            sl = slice(int(offsets[i]), int(offsets[i] + lengths[i]))
            a0, g0 = starts[i]
            lli, ai, gi = self._profile_one(ycat[sl], tcat[sl], p, pi0, a0, g0)
            ll += lli
            a_hat[i] = ai
            g_hat[i] = gi
        self._ag_cache = list(zip(a_hat.tolist(), g_hat.tolist()))
        return ll, a_hat, g_hat, p

    def _nll(self, theta, packed):
        ycat, tcat, lengths, offsets = packed
        p = self._unpack(theta)
        pi0 = _stationary_probs(p["P"]).astype(np.float64)
        if not self.country_intercepts and not self.country_trends:
            zcat = ycat - p["a"] - p["g"] * tcat
            ll = _panel_ll_nb(
                zcat, lengths, offsets,
                np.ascontiguousarray(p["rho"], dtype=np.float64),
                np.ascontiguousarray(p["mu"], dtype=np.float64),
                np.ascontiguousarray(p["sigma"], dtype=np.float64),
                np.ascontiguousarray(p["P"], dtype=np.float64),
                pi0,
            )
        else:
            ll, _, _, _ = self._profile_all(theta, packed)
        if not np.isfinite(ll):
            return 1e12
        return -float(ll)

    def _starting_values(self, panels, n_starts, rng):
        ys = np.concatenate([y for y, _ in panels])
        ts = np.concatenate([t for _, t in panels])
        X = np.column_stack([np.ones(len(ys)), ts])
        beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
        a0, g0 = float(beta[0]), float(beta[1])
        if self.country_intercepts or self.country_trends:
            pieces = []
            for y, t in panels:
                ai, gi = self._ols_ag(y, t)
                a_use = ai if self.country_intercepts else a0
                g_use = gi if self.country_trends else g0
                pieces.append(y - a_use - g_use * t)
            resid = np.concatenate(pieces)
        else:
            resid = ys - a0 - g0 * ts
        s_hat = float(np.std(resid, ddof=1)) or 0.05

        rhos = []
        for y, t in panels:
            ai, gi = self._ols_ag(y, t)
            a_use = ai if self.country_intercepts else a0
            g_use = gi if self.country_trends else g0
            z = y - a_use - g_use * t
            if z.size < 4:
                continue
            zc = z - z.mean()
            den = float(np.dot(zc[:-1], zc[:-1]))
            if den <= 0:
                continue
            rhos.append(float(np.dot(zc[1:], zc[:-1]) / den))
        if rhos:
            rho0 = float(np.clip(np.median(rhos), 0.2, 0.95))
        else:
            rho0 = 0.7

        k = self.n_regimes
        mid = self._mid_regime()
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
            for j in range(1, mid + 1):
                mu[mid - j] = -j * spread
                mu[mid + j] = j * spread
            return mu

        def one(g, a, rho, mu_spread, sigs, P, jitter=0.0):
            mu = spread_mu(mu_spread)
            th = self._pack_from_dicts(P, np.full(k, rho), mu, np.asarray(sigs, float), g, a)
            if jitter:
                th = th + rng.normal(0.0, jitter, size=th.shape)
            return th

        dist = np.abs(np.arange(k) - mid)
        scale = dist / max(mid, 1)
        sig_het = s_hat * (0.8 + 0.4 * scale)
        sig_base = np.full(k, s_hat)

        starts = [
            one(g0, a0, rho0, s_hat, sig_het, P0),
            one(g0, a0, 0.5, 0.5 * s_hat, sig_base, P0),
            one(g0, a0, 0.85, 1.5 * s_hat, sig_het, P0),
            one(0.0, float(np.mean(ys)), 0.6, s_hat, sig_base, P0),
        ]
        while len(starts) < n_starts:
            starts.append(
                one(
                    g0 + rng.normal(0, abs(g0) * 0.25 + 0.005),
                    a0 + rng.normal(0, s_hat),
                    rng.uniform(0.3, 0.9),
                    abs(rng.normal(0, s_hat)),
                    np.maximum(sig_het * rng.uniform(0.7, 1.4, size=k), 1e-4),
                    P0,
                    jitter=0.08,
                )
            )
        return starts[:n_starts]

    def _order_regimes(self, theta):
        """Permute so mu is increasing, then shift so the pinned mean is 0."""
        p = self._unpack(theta)
        k = self.n_regimes
        order = np.argsort(p["mu"])
        mu = p["mu"][order]
        rho = p["rho"][order]
        sig = p["sigma"][order]
        P = p["P"][np.ix_(order, order)]
        a = p["a"]
        g = p["g"]
        pin = self._mid_regime()
        shift = mu[pin]
        mu = mu - shift
        a = a + shift
        return self._pack_from_dicts(P, rho, mu, sig, g, a)

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
        if "a" in raw:
            out["a"] = raw["a"]
        if "g" in raw:
            out["g"] = raw["g"]

        if not self.common_rho:
            out["rho"] = np.array([
                raw[f"rho[{s}]"] * (1.0 - p["rho"][s] ** 2) for s in range(k)
            ])
        else:
            r = float(p["rho"][0])
            out["rho"] = raw["rho"] * (1.0 - r ** 2)

        mu_se = np.full(k, np.nan)
        mu_se[self._mid_regime()] = 0.0
        for s in self._free_mu_indices():
            mu_se[s] = raw[f"mu[{s}]"]
        out["mu"] = mu_se

        if not self.common_sigma:
            out["sigma"] = np.array([
                raw[f"sigma[{s}]"] * float(p["sigma"][s]) for s in range(k)
            ])
        else:
            out["sigma"] = raw["sigma"] * float(p["sigma"][0])
        out["P"] = self._se_P(p["P"], cov)
        return out

    def fit(
        self,
        country,
        time,
        y,
        n_starts=8,
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
            # Still proceed; _prepare warnings will carry this too.

        panels, ids, t0, info = self._prepare(country, time, y)
        warnings = list(info["warnings"])
        if not HAS_NUMBA:
            warnings.insert(0, (
                "numba is not installed; likelihood evaluations use pure Python."
            ))
        if self.n_regimes == 1:
            warnings.append(
                "n_regimes=1 is a non-switching AR(1) around the common trend "
                "(the single mean is pinned at 0)."
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
        packed = self._stack_panels(panels)
        self._ols = [self._ols_ag(y, t) for y, t in panels]
        self._ag_cache = None
        _warmup_numba()
        starts = self._starting_values(panels, n_starts, rng)
        best, best_fun, best_msg, best_ok = None, np.inf, "", False
        n_ok = 0
        for i, th0 in enumerate(starts):
            opt = minimize(
                self._nll,
                th0,
                args=(packed,),
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-8},
            )
            if opt.success:
                n_ok += 1
            if verbose:
                print(
                    f"  start {i + 1}/{n_starts}: nll={opt.fun:.4f}  "
                    f"success={opt.success}  {opt.message}",
                    flush=True,
                )
            if opt.fun < best_fun:
                best_fun = float(opt.fun)
                best = opt.x.copy()
                best_msg = str(opt.message)
                best_ok = bool(opt.success)

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
        _, a_hat, g_hat, p = self._profile_all(best, packed)
        if compute_se:
            stderr, cov = self._stderr(best, packed)
            se_params = self._se_transformed(best, stderr, cov)
        else:
            stderr, se_params = None, None
        if compute_se and stderr is not None and not np.any(np.isfinite(stderr)):
            warnings.append(
                "Numerical Hessian could not be inverted; std errors are missing. "
                "For a paper, bootstrap countries."
            )

        mu = np.asarray(p["mu"])
        k = self.n_regimes
        if k >= 3:
            gaps = np.diff(mu)
            free_near_zero = np.any(np.abs(mu[self._free_mu_indices()]) < 1e-3)
            if free_near_zero or np.any(gaps < 1e-3):
                warnings.append(
                    "At least two regime means collapsed (gap < 1e-3 or a "
                    "free mean near 0). Coefficient estimates for those "
                    "regimes may not be separately identified."
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
            for i, (cid, (yy, tt)) in enumerate(zip(ids, panels)):
                z = yy - a_hat[i] - g_hat[i] * tt
                _, filt = _country_loglik(
                    z, p["rho"], p["mu"], p["sigma"], p["P"], pi0, return_filter=True
                )
                out = {"time": tt + t0, "cycle": z}
                for s in range(self.n_regimes):
                    out[f"p_regime{s}"] = filt[:, s]
                filtered[cid] = pd.DataFrame(out)

        rho_out = float(p["rho"][0]) if self.common_rho else p["rho"]
        sig_out = float(p["sigma"][0]) if self.common_sigma else p["sigma"]
        if self.country_intercepts:
            a_out = {cid: float(a_hat[i]) for i, cid in enumerate(ids)}
        else:
            a_out = float(a_hat[0])
        if self.country_trends:
            g_out = {cid: float(g_hat[i]) for i, cid in enumerate(ids)}
        else:
            g_out = float(g_hat[0])
        params = {
            "P": p["P"],
            "rho": rho_out,
            "mu": p["mu"],
            "sigma": sig_out,
            "g": g_out,
            "a": a_out,
        }

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
            common_sigma=self.common_sigma,
            country_intercepts=self.country_intercepts,
            country_trends=self.country_trends,
        )
        if detrend_pdf:
            self.res_.plot_detrended(detrend_pdf)
        return self.res_

    def _stderr(self, theta, panels):
        theta = np.asarray(theta, dtype=float)
        n = theta.size
        eps = 1e-4 * (1.0 + np.abs(theta))
        f0 = self._nll(theta, panels)
        H = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = eps[i]
            for j in range(i, n):
                ej = np.zeros(n)
                ej[j] = eps[j]
                if i == j:
                    fp = self._nll(theta + ei, panels)
                    fm = self._nll(theta - ei, panels)
                    H[i, i] = (fp - 2.0 * f0 + fm) / (eps[i] ** 2)
                else:
                    fpp = self._nll(theta + ei + ej, panels)
                    fpm = self._nll(theta + ei - ej, panels)
                    fmp = self._nll(theta - ei + ej, panels)
                    fmm = self._nll(theta - ei - ej, panels)
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
    """Draw an unbalanced panel from the model.

    time_step is the calendar spacing of consecutive observations (1 = annual
    if year0 is a calendar year; 0.25 = quarterly on a yearly scale). rho is
    per observation; g is per unit of calendar time.
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
            z = mu[s] * (1.0 - rho_v[s]) + rho_v[s] * z + sigma[s] * rng.normal()
            s = int(rng.choice(k, p=P[s]))
            rows.append((i, cal[h], a + g * t[h] + z, s, z))
    df = pd.DataFrame(rows, columns=["country", "time", "y", "s", "z"])
    df["year"] = df["time"]
    return df
