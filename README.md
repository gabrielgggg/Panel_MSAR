# Panel MS-AR(1) after a pooled linear trend

Two-step estimator for an unbalanced country panel. Step 1 is pooled OLS of \(y\) on a constant and calendar time. Step 2 is a Markov-switching AR(1) on that residual. Countries are independent given the shared Markov parameters. Latent regimes are country-specific.

## Model

$$
\begin{aligned}
e_{it} &= y_{it} - a - g\, t, \\
s_{i,t+1} &\sim \Pi(\,\cdot\mid s_{it}), \\
z_{i,t+1} &= \bigl(1-\rho(s_{i,t+1})\bigr)\mu(s_{i,t+1}) + \rho(s_{i,t+1})\, z_{it} + \sigma(s_{i,t+1})\,\varepsilon_{it}.
\end{aligned}
$$

\(a\) and \(g\) are the pooled OLS coefficients on the estimation sample, so \(\sum e_{it}=0\) and \(\sum e_{it}\, t_{it}=0\). The reported cycle is \(e_{it}\). The MS-AR does not estimate \(a\) or \(g\). \(E[z]=0\) pins the level of the cycle inside the likelihood: one regime mean is solved from that restriction, and regimes are then ordered by \(\mu\).

The regime dated \(t\) is the one that produced \(z_t\). The next regime is drawn first; \(z_{t+1}\) then steps under that new regime.

| Piece | Specification |
|---|---|
| Outcome \(y\) | Any series; logs are the intended scale. |
| Trend | One pooled intercept \(a\) and one pooled slope \(g\), removed before the likelihood. \(a\) is the intercept at the first sample date. \(g\) is per unit of calendar time. No Hessian standard errors. |
| Time \(t\) | Calendar time, **common origin** for every country. Any regular frequency. |
| Regimes \(k\) | Chosen by the user (\(k\ge 1\)). \(k\) is specified, not selected. |
| Mean restriction | \(E[z]=0\) for the switching process. |
| Persistence | One \(\rho\) for all regimes (`common_rho=True`) or switching \(\rho(s)\). \(|\rho|<\texttt{rho\_max}\) (default 0.995). |
| Variance | \(\sigma(s)\) switches with the regime. |
| Transitions | Common \(\Pi\). Latent path \(s_{it}\) is country-specific. |
| Initial condition | \(z_1 \mid s_1 \sim N\bigl(\mu_s,\, \sigma_s^2/(1-\rho_s^2)\bigr)\), with \(s_1\) from the ergodic distribution of \(\Pi\). |
| Sample | Unbalanced panel. Drop countries with fewer than `min_t` **observations**. Calendar gaps: keep the longest contiguous spell (no interpolation). |

The OLS step zeros the observation-weighted pooled slope. It does not zero each country's own slope.

## Estimator

`panel_msar.py` (`PanelMSAR`). Multi-start L-BFGS-B on unconstrained cycle parameters: row-wise softmax logits for \(\Pi\) (`k(k-1)` free); `rho = rho_max * tanh`; `sigma = exp`; free means are every `mu[s]` except the one solved from \(E[z]=0\). Starts run in a process pool. Numba Hamilton filter (install `numba`). Hessian standard errors are on the unconstrained cycle parameters, then delta-method to the table. \(a\) and \(g\) are reported from the OLS step only.

`demo_panel_msar.py` simulates a 3-regime panel and fits this procedure.

`msar_report.py` holds LaTeX helpers (`compile_tex`, cycle plots, tables). Compile TeX from the `.tex` parent directory.

`fortran/NL.f90` discretizes a stationary MS-AR cycle (Farmer–Toda on a common \(z\) grid, joint \((z,s)\) chain) with the same timing: draw \(s'\) first, then step \(z\) under \(s'\).

## Data and API

Pass `country`, `time`, `y` (ideally in logs). `time` must share one calendar origin, not periods-since-entry.

- Year-fraction, e.g. quarterly `1970.0, 1970.25, …`: `g` per year, `rho` per quarter.
- Datetimes / pandas `Period`s: converted to `year + (month-1)/12`.
- Integer period index (Stata `%tq`): left as-is; `g` per period.
- Do not pass year.quarter codes (`1970.1, 1970.2, 1970.3, 1970.4`).

Internally `t` is shifted so the earliest sample date is 0 (`res.time_base`, `res.time_step`). OLS \(a\) is the intercept on that shifted scale.

```python
from panel_msar import PanelMSAR

mod = PanelMSAR(n_regimes=3, common_rho=True, min_t=12, rho_max=0.99)
res = mod.fit(
    df["country"], df["time"], df["y"],
    n_starts=7, maxiter=400, detrend_pdf="cycle.pdf",
)
print(res)                 # OLS a, g; rho; regime table; Pi
res.params                 # a, g, rho, mu, sigma, P, pi, Ez
res.filtered_probs[cid]    # time, cycle (the OLS residual), p_regime0/1/2
res.plot_detrended("cycle.pdf")
```

Current empirical extract: `data_2026.09.17/` (seasonally adjusted real GDP per worker). The 1985–2019 runner is `python data_2026.09.17/run_msar_common_1985.py`. The checked-in PDF for that window was estimated before this two-step baseline; running the script again replaces it.

Dependencies: `numpy`, `pandas`, `scipy`, `numba`, `matplotlib`.
