# Panel MS-AR(1) around a common log-linear trend

Joint MLE of one model on an unbalanced country panel. Shared Markov parameters; country-specific latent regimes. Optional Normal random intercepts. `y` can be any series; logs are the intended scale.

## Model

$$
\begin{aligned}
y_{it} &= a + g\, t + z_{it}, \\
z_{i,t+1} &= \bigl(1-\rho(s_{it})\bigr)\mu(s_{it}) + \rho(s_{it})\, z_{it} + \sigma(s_{it})\,\varepsilon_{it}, \\
s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}).
\end{aligned}
$$

With `random_intercepts=True`, \(a\) is replaced by \(a_i\sim N(\alpha,\omega^2)\). With `convergence=True`, the permanent intercept is replaced by a decaying gap \(b_i\lambda^{t-T_{i0}}\) (\(b_i\sim N(0,\omega_b^2)\), \(0<\lambda<1\) common; \(T_{i0}\) is country \(i\)'s first observation). The slope \(g\) is always common. `convergence=True` forces random intercepts.

Countries are independent given shared \(\theta\). Likelihood: Hamilton filter per country, sum log-likelihoods. Random intercepts are integrated with 11-point Gauss–Hermite quadrature.

| Piece | Specification |
|---|---|
| Outcome \(y\) | Any series; ideally in logs. |
| Trend | Common intercept \(a\) and common slope \(g\). \(g\) is per unit of calendar time; \(\rho\) is per observation. |
| Time \(t\) | Calendar time, **common origin** for every country. Any regular frequency. |
| Random intercepts | Off by default. If on, \(a_i\sim N(\alpha,\omega^2)\); \(\alpha\) and \(\omega\) are MLE parameters. Cycles use posterior-mean \(a_i\). |
| Catch-up | Off by default (`convergence=False`). If on, \(y_{it}=\bar a + g t + b_i\lambda^{t-T_{i0}}+z_{it}\). |
| Regimes \(k\) | Odd (\(1, 3, 5, \ldots\)) so a unique median regime exists. \(k\) is specified, not selected. |
| Mean restriction | After estimation, regimes are ordered by \(\mu\) and shifted so \(\mu_{\lfloor k/2\rfloor}=0\). The shift is absorbed into \(a\) (or \(\alpha\)). Set `zero_mu=True` to restrict **every** \(\mu(s)=0\); regimes are then ordered by \(\sigma\) (or \(\rho\) if \(\sigma\) is common). |
| Persistence | One \(\rho\) for all regimes (`common_rho=True`) or switching \(\rho(s)\). |
| Variance | \(\sigma(s)\) switches with the regime (`common_sigma=False`) or is common. |
| Transitions | Common \(\Pi\). Latent path \(s_{it}\) is country-specific. |
| Timing | Regime dated \(t\) governs the transition from \(z_t\) to \(z_{t+1}\); then a new regime is drawn. |
| Initial condition | \(z_1 \mid s_1 \sim N\bigl(\mu_s,\, \sigma_s^2/(1-\rho_s^2)\bigr)\), with \(s_1\) from the ergodic distribution of \(\Pi\). |
| Sample | Unbalanced panel. Drop countries with fewer than `min_t` **observations**. Calendar gaps: keep the longest contiguous spell (no interpolation). |

\(a\) (or \(\alpha\)) and the regime means are collinear without the median-mean pin.

## Estimator

`panel_msar.py` (`PanelMSAR`). Multi-start L-BFGS-B on unconstrained parameters: row-wise softmax logits for `Pi` (`k(k-1)` free); `rho = rho_max * tanh`; `sigma = exp`; free means are every `mu[s]` except `s = k//2`, or none if `zero_mu=True`. After each successful fit, regimes are ordered by `mu` and the median mean is shifted into `a` / `alpha` (or ordered by `sigma` if all means are zero). Numba Hamilton filter (install `numba`). Hessian SEs are on the unconstrained vector, then delta-method to the table.

`demo_panel_msar.py` simulates a 3-regime DGP and recovers parameters.

`msar_report.py` holds LaTeX helpers (`compile_tex`, cycle plots, tables). Compile TeX from the `.tex` parent directory.

`fortran/NL.f90` discretizes the stationary MS-AR cycle (Farmer–Toda on a common \(z\) grid, joint \((z,s)\) chain).

## Data and API

Pass `country`, `time`, `y` (ideally in logs). `time` must share one calendar origin, not periods-since-entry.

- Year-fraction, e.g. quarterly `1970.0, 1970.25, …`: `g` per year, `rho` per quarter.
- Datetimes / pandas `Period`s: converted to `year + (month-1)/12`.
- Integer period index (Stata `%tq`): left as-is; `g` per period.
- Do not pass year.quarter codes (`1970.1, 1970.2, 1970.3, 1970.4`).

Internally `t` is shifted so the earliest sample date is 0 (`res.time_base`, `res.time_step`).

```python
from panel_msar import PanelMSAR

mod = PanelMSAR(
    n_regimes=3, common_rho=True, common_sigma=False,
    random_intercepts=False, convergence=False,
    zero_mu=False, min_t=12, rho_max=0.99,
)
res = mod.fit(
    df["country"], df["time"], df["y"],
    n_starts=7, maxiter=400, detrend_pdf="cycle.pdf",
)
print(res)                 # a, g, rho, regime table, Pi; SEs in parentheses underneath
res.params                 # a, g, rho, mu, sigma, P (and alpha, omega if RE)
res.filtered_probs[cid]    # time, cycle, p_regime0/1/2
res.plot_detrended("cycle.pdf")
```

Current empirical extract: `data_2026.09.17/` (seasonally adjusted real GDP per worker). Runner: `python data_2026.09.17/run_msar_re.py`.

Dependencies: `numpy`, `pandas`, `scipy`, `numba`, `matplotlib`.
