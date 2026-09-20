"""Univariate local-level (random-walk-plus-noise) trend for a common tau_t."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def common_growth_trend(country, time, y):
    """Common tau_t from average growth of continuing countries.

    At each date t, Delta tau is the mean of y_it - y_i,t- over countries
    observed at both t and the previous grid date. New entrants affect tau
    only through later growth, not through their level, so poorer countries
    entering later do not pull tau down. tau is a cumulated common RW.
    The level is pinned so tau at t0 equals the mean of a balanced core
    (countries present at every date), else the median at t0.
    """
    country = np.asarray(country)
    time = np.asarray(time, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(time) & np.isfinite(y)
    country, time, y = country[ok], time[ok], y[ok]
    grid = np.unique(time)
    grid.sort()
    lookup = {}
    for c, t, yi in zip(country, time, y):
        lookup[(c, float(t))] = float(yi)
    present = {float(t): set(country[time == t]) for t in grid}
    dtau = np.zeros(grid.size)
    n_d = np.zeros(grid.size, dtype=int)
    for i in range(1, grid.size):
        t0, t1 = float(grid[i - 1]), float(grid[i])
        both = present[t0] & present[t1]
        if not both:
            dtau[i] = dtau[i - 1] if i else 0.0
            continue
        diffs = [lookup[(c, t1)] - lookup[(c, t0)] for c in both]
        dtau[i] = float(np.mean(diffs))
        n_d[i] = len(diffs)
    tau = np.cumsum(dtau)
    core = set.intersection(*present.values()) if present else set()
    t0 = float(grid[0])
    if core:
        pin = float(np.mean([lookup[(c, t0)] for c in core if (c, t0) in lookup]))
    else:
        pin = float(np.median([lookup[(c, t0)] for c in present[t0]]))
    tau = tau - tau[0] + pin
    ybar = np.array([
        float(np.mean([lookup[(c, float(t))] for c in present[float(t)]]))
        for t in grid
    ])
    n = np.array([len(present[float(t)]) for t in grid], dtype=int)
    return {
        "times": grid,
        "tau": tau,
        "ybar": ybar,
        "n": n,
        "n_delta": n_d,
        "mu": float(np.mean(dtau[1:])) if grid.size > 1 else 0.0,
        "n_core": len(core),
    }


def unbalanced_cs_mean(time, y, min_n=1):
    """Cross-sectional mean of y at each calendar time (unbalanced)."""
    time = np.asarray(time, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.unique(time[np.isfinite(time)])
    grid.sort()
    mean = np.full(grid.size, np.nan)
    n = np.zeros(grid.size, dtype=int)
    for i, t in enumerate(grid):
        sl = (time == t) & np.isfinite(y)
        n[i] = int(sl.sum())
        if n[i] >= min_n:
            mean[i] = float(np.mean(y[sl]))
    return grid, mean, n


def _filter_ll(y, mu, se2, sn2):
    n = y.size
    a = np.empty(n)
    P = np.empty(n)
    a_pred = np.empty(n)
    P_pred = np.empty(n)
    ll = 0.0
    first = True
    for t in range(n):
        if first:
            a_pred[t] = y[t] if np.isfinite(y[t]) else 0.0
            P_pred[t] = 1.0e4
            first = False
        else:
            a_pred[t] = a[t - 1] + mu
            P_pred[t] = P[t - 1] + sn2
        if not np.isfinite(y[t]):
            a[t] = a_pred[t]
            P[t] = P_pred[t]
            continue
        F = P_pred[t] + se2
        if F <= 1e-12:
            F = 1e-12
        v = y[t] - a_pred[t]
        K = P_pred[t] / F
        a[t] = a_pred[t] + K * v
        P[t] = max((1.0 - K) * P_pred[t], 1e-12)
        ll += -0.5 * (np.log(2.0 * np.pi) + np.log(F) + v * v / F)
    return ll, a, P, a_pred, P_pred


def _smooth(a, P, a_pred, P_pred, mu, sn2):
    n = a.size
    tau = np.empty(n)
    tau[-1] = a[-1]
    for t in range(n - 2, -1, -1):
        Pp = P_pred[t + 1]
        J = 0.0 if Pp <= 1e-12 else P[t] / Pp
        tau[t] = a[t] + J * (tau[t + 1] - a_pred[t + 1])
    return tau


def fit_local_level(y, times=None):
    """MLE local level: y_t = tau_t + e_t, tau_t = tau_{t-1} + mu + eta_t."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 8:
        raise ValueError("Need at least 8 observations for the local-level trend.")

    def nll(th):
        mu, log_se, log_sn = th
        se2 = float(np.exp(2.0 * np.clip(log_se, -12.0, 5.0)))
        sn2 = float(np.exp(2.0 * np.clip(log_sn, -12.0, 5.0)))
        ll, *_ = _filter_ll(y, mu, se2, sn2)
        return -ll if np.isfinite(ll) else 1e12

    y_ok = y[ok]
    d = np.diff(y_ok)
    mu0 = float(np.mean(d)) if d.size else 0.0
    s0 = float(np.std(d, ddof=1) or 0.05)
    th0 = np.array([mu0, np.log(max(s0, 1e-4)), np.log(max(0.5 * s0, 1e-4))])
    opt = minimize(nll, th0, method="L-BFGS-B")
    mu, log_se, log_sn = opt.x
    se2 = float(np.exp(2.0 * np.clip(log_se, -12.0, 5.0)))
    sn2 = float(np.exp(2.0 * np.clip(log_sn, -12.0, 5.0)))
    ll, a, P, a_pred, P_pred = _filter_ll(y, mu, se2, sn2)
    tau = _smooth(a, P, a_pred, P_pred, mu, sn2)
    return {
        "tau": tau,
        "mu": float(mu),
        "sigma_e": float(np.sqrt(se2)),
        "sigma_eta": float(np.sqrt(sn2)),
        "loglik": float(ll),
        "success": bool(opt.success),
        "message": str(opt.message),
        "y": y,
        "times": None if times is None else np.asarray(times, dtype=float),
    }
